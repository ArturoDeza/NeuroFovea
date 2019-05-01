require 'torch'
require 'unsup'
require 'nn'
require 'image'
require 'paths'
require 'lib/AdaptiveInstanceNormalization'
require 'lib/utils'
require 'nngraph'
util = dofile('util/util.lua')

-----------------------------------------
-- Code Modified by Arturo Deza.       -- 
-- Based on Original Adaptive Instance -- 
-- Normalization code by Xun Huang.    -- 
-----------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-image','','File path to original image that we want to metamerize')
cmd:option('-imageDir', '', 'Directory path to a batch of content images')
cmd:option('-vgg', 'models/vgg_normalised.t7', 'Path to the VGG network')
cmd:option('-decoder', 'models/decoder-content-similar.t7', 'Path to the decoder')

-- Additional options
cmd:option('-imageSize',512,'input Image Size')
cmd:option('-crop', false, 'If true, center crop both content and style image before resizing')
cmd:option('-saveExt', 'png', 'The extension name of the output image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-outputDir', 'output_Stimuli', 'Directory to save the output image(s)')

-- Advanced options
cmd:option('-styleInterpWeights', '', 'The weight for blending the style of multiple style images')
cmd:option('-mask', '', 'Mask to apply spatial control, assume to be the path to a binary mask of the same size as content image')
cmd:option('-scale','0.25','Receptive Field rate of growth')

cmd:option('-refinement',0,'Flag to use superresolution refinement module.')
cmd:option('-color',0,'Flag to use the color refinement module')
cmd:option('-reference',0,'Flag to compute the reference image. If one alpha values will be computed via the gamma function as a function of receptive field size, otherwise all alpha values will be set to zero.')

-- Parse arguments:
opt = cmd:parse(arg)
-- Set the style and content image as the same
opt.style = opt.image
opt.content = opt.image

assert(opt.content ~= '' or opt.imageDir ~= '', 'Either --content or --imageDir should be given.')
assert(opt.content == '' or opt.imageDir == '', '--image and --imageDir cannot both be given.')
assert(paths.filep(opt.decoder), 'Decoder ' .. opt.decoder .. ' does not exist.')


print(opt)
if opt.gpu >= 0 then
    require 'cudnn'
    require 'cunn'
end

if opt.refinement == 1 then
	if opt.color == 1 then
		netG = torch.load('./Refinement/latest_net_G.t7')
	elseif opt.color == 0 then
		netG = torch.load('./Refinement/200_net_G.t7')
	end
end

scale = opt.scale
peripheral_windows = 'Receptive_Fields/MetaWindows_clean_s' .. scale .. '/'
pooling_address = paths.dir(peripheral_windows)
num_pooling_regions = #pooling_address - 3

-- This parameter is the one we found that fitted the 
-- sigmoid described in the paper for tuning the alpha values 
-- given the size of each receptive field.
d = 0.56

mask_matrix = {}

if opt.mask == '' then
		for i=0,num_pooling_regions do
			local mask_src = peripheral_windows .. tostring(i) .. '.png'
			local maskx_temp = image.load(mask_src,1,'float')
			table.insert(mask_matrix,maskx_temp)
		end
end

-----------------------------------------------------------------------------------------
-- Construct Alpha Values:
-- If you want to construct your own gamma function or vary the content to texture ratio per 
-- pooling regiona manually (this is needed when you change the point of fixation from the center),
-- you will have to modify these lines of code below whih hold all the alpha[i] values per pooling region. 
-- Note that when you change the point of fixation, the circular symmetry changes,
-- so you might have to write another program or script that verifies that you are indeed 
-- assigning the correct alpha value to the matching pooling region.

-- Alpha Values for Image-to-Texture Localized Style Transfer for every pooling region.
-- This function is parametrized by the gamma function, but any values can be used as input 
-- to simulate different losses in the visual field.

alpha_matrix = '0.0,'
for k=1,num_pooling_regions do

	fgmask = mask_matrix[k+1]:gt(0.5):sum()
	recep_size = torch.sqrt(fgmask/3.14)*26.0/512.0

	alpha_value = -1.0 + 2.0/(1+torch.exp(-torch.log(10)*recep_size*d))
	alpha_matrix = alpha_matrix .. alpha_value

	if k<num_pooling_regions then
		alpha_matrix = alpha_matrix .. ','
	end
end		

alpha = alpha_matrix:split(',')
for i=1,#alpha do
	alpha[i] = alpha[i]/1.0
end

----------------------------------------------------------------------------------------------

-- Load the last layer of VGG to get up to relu4_1
vgg = torch.load(opt.vgg)
for i=53,32,-1 do
    vgg:remove(i)
end

-- Load Adaptive Instance Normalization Module:
local adain = nn.AdaptiveInstanceNormalization(vgg:get(#vgg-1).nOutputPlane)
decoder = torch.load(opt.decoder)

if opt.gpu >= 0 then
    cutorch.setDevice(opt.gpu+1)
    vgg = cudnn.convert(vgg, cudnn):cuda()
    adain:cuda()
    decoder:cuda()
else
    vgg:float()
    adain:float()
    decoder:float()
end

local function styleTransfer(content, style, noise)

		-- Create Noise image
		local noise = torch.randn(3,512,512)
		noise = coral(noise:double(), content:double())

		local contentImg = content
		noise = sizePreprocess(noise, opt.crop, opt.imageSize)

    if opt.gpu >= 0 then
        content = content:cuda()
        style = style:cuda()
				noise = noise:cuda()
    else
        content = content:float()
        style = style:float()
				noise = noise:float()
    end

    styleFeature = vgg:forward(style):clone()
    contentFeature = vgg:forward(content):clone()
		noiseFeature = vgg:forward(noise):clone()

		-- Spatial Control
		-- Get Number of Channels, Heigh and Width of the features after the forward pass through the Encoder.
    local C, H, W = contentFeature:size(1), contentFeature:size(2), contentFeature:size(3)

		-- Initialize target feature to zero.
		targetFeature = contentFeature:view(C,-1):clone():zero()

		-- Loop over all receptive fields and perform localized auto-style transfer: 
		for j=0,num_pooling_regions do
			local styleFeatureM = styleFeature

			local maskResized = image.scale(mask_matrix[j+1],W,H,'simple')
			local maskView = maskResized:view(-1)

			local fgmask = torch.LongTensor(torch.find(maskView:gt(0.001),1))

   		local contentFeatureView = contentFeature:view(C, -1)
			local contentFeatureM = contentFeatureView:index(2,fgmask):view(C,fgmask:nElement(),1)

			local styleFeatureView = styleFeatureM:view(C,-1)
			local styleFeatureM2 = styleFeatureView:index(2,fgmask):view(C,fgmask:nElement(),1)

			local noiseFeatureView = noiseFeature:view(C,-1)
			local noiseFeatureM = noiseFeatureView:index(2,fgmask):view(C,fgmask:nElement(),1)

			-- Perform style transfer on noise.
			local noiseFeatureM3 = adain:forward({noiseFeatureM,styleFeatureM2}):clone():squeeze()

			-- Perform Local content to texture interpolation for each receptive field
			targetFeatureM = alpha[j+1]*noiseFeatureM3 + (1-alpha[j+1])*contentFeatureM -- original!

			targetFeature:indexCopy(2,fgmask,targetFeatureM)
		end

    targetFeature = targetFeature:viewAs(contentFeature)
    targetFeature = targetFeature:squeeze()

		-- Decode only one vector
    return decoder:forward(targetFeature) 
end

print('Creating save folder at ' .. opt.outputDir)
paths.mkdir(opt.outputDir)

local imagePaths = {}

if opt.image ~= '' then -- use a single input image
    table.insert(imagePaths, opt.image)
else -- use a batch of input images to metamerize
    assert(opt.imageDir ~= '', "Either opt.imageDir or opt.image should be non-empty!")
    imagePaths = extractImageNamesRecursive(opt.imageDir)
end

local numImage = #imagePaths
print("# Input images: " .. numImage)

for i=1,numImage do
    local imagePath = imagePaths[i]
    local imageExt = paths.extname(imagePath)
    local imageImg = image.load(imagePath, 3, 'float') -- ranges from 0 to 1
    local imageName = paths.basename(imagePath, imageExt)
    local imageImg = sizePreprocess(imageImg, opt.crop, opt.imageSize)

		noiseImg = torch.randn(3,512,512)
		
		-- Note that here the style and content image are the same input image.
		output = styleTransfer(imageImg, imageImg, noiseImg)

		output = output:float():add_dummy()

		if opt.refinement == 1 then
			output:clamp(0.0,1.0)
			output = netG:forward(util.preprocess_batch(output:cuda()))
			output = util.deprocess_batch(output)
		end
		output = output:squeeze()
		metamer = output

		local savePath = paths.concat(opt.outputDir, imageName .. '_metamer_s' .. scale .. '.' .. opt.saveExt)
		print('Output image saved at: ' .. savePath)
		image.save(savePath, metamer)
		print('Success!')

end
