from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer
import json

def model1(checkpoint): ## 除text_encoder外都冻结
    text_encoder = CLIPTextModel.from_pretrained(checkpoint,
                                                 subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder='vae')
    model_name = checkpoint = 'CompVis/stable-diffusion-v1-4'
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
    #unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder='unet')

    text_encoder.train()
    vae.eval()
    unet.eval()
    tokenizer = CLIPTokenizer.from_pretrained(
    checkpoint,
    subfolder='tokenizer',
    )
    #添加新词
    text_encoder.resize_token_embeddings(tokenizer.vocab_size + 37)
    
    #初始化新词的参数 toy -> <cat-toy>
    token_embeds = text_encoder.get_input_embeddings().weight.data
    
    for i in range(49408,49408+ 37):
        token_embeds[i] = token_embeds[10738]

    #冻结参数
    for param in vae.parameters():
        param.requires_grad = False

    for param in unet.parameters():
        param.requires_grad = False

    for name, param in text_encoder.named_parameters():
        #除了这一层,其他全部冻结
        if name != 'text_model.embeddings.token_embedding.weight':
            param.requires_grad = False

    return text_encoder, vae, unet


def model2(checkpoint): ## 除text_encoder外都冻结
    text_encoder = CLIPTextModel.from_pretrained(checkpoint,
                                                 subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet')

    text_encoder.train()
    vae.eval()
    unet.eval()
    
    
    #初始化新词的参数 toy -> <cat-toy>
    token_embeds = text_encoder.get_input_embeddings().weight.data
    
    cheese_list_less20_directory = './list_of_cheese_less20.txt'
    f2 = open('./models/cheese_chellenge/tokenizer/added_tokens' + '.json', 'r')
    dic = json.load(f2)
    f2.close()
    f = open(cheese_list_less20_directory)
    cheese_list_less20 = []
    for name in f:
        cheese_list_less20.append('<'+ name.replace('\n','') + '>')
    
    cheese_list_less20_array = []
    for name in cheese_list_less20:
        cheese_list_less20_array.append(dic[name])
    
    for i in cheese_list_less20_array:
        token_embeds[i] = token_embeds[10738]
        
    #冻结参数
    for param in vae.parameters():
        param.requires_grad = False

    for param in unet.parameters():
        param.requires_grad = False

    for name, param in text_encoder.named_parameters():
        #除了这一层,其他全部冻结
        if name != 'text_model.embeddings.token_embedding.weight':
            param.requires_grad = False

    return text_encoder, vae, unet


