from model import Net,Vgg
from imgLoad import loadImg
import torchvision
from torchvision import transforms
import torch
import argparse


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# devcice = torch.device('cpu')
def main(config):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])
    
    contentImage = loadImg(config.content, transform)
    styleImage = loadImg(config.style, transform, (contentImage.size(2), contentImage.size(3)))
    target = contentImage.clone()

    net = Net(target.size(1)).to(dev)
    vgg = Vgg().to(dev)
    
    optimizer = torch.optim.Adam([{"params":net.parameters()}], lr=config.lr, betas=[0.5, 0.999])
    
    for step in range(config.epoch):
        contentFeatures = vgg(contentImage)
        styleFeatures = vgg(styleImage)
        target = net(target)
        targetFeatures = vgg(target)
        styleLoss = 0
        contentLoss = 0
        for f1, f2, f3 in zip(targetFeatures, contentFeatures, styleFeatures):
            # Compute content loss with target and content images
            contentLoss += torch.mean((f1 - f2)**2)

            # Reshape convolutional feature maps
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            # Compute gram matrix
            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            # Compute style loss with target and style images
            styleLoss += torch.mean((f1 - f3)**2) / (c * h * w)
        
        # 计算总体损失
        loss = contentLoss + config.styleWeight * styleLoss 
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        if (step+1) % 10 == 0:
            print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}' 
                   .format(step+1, config.epoch, contentLoss.item(), styleLoss.item()))

        if (step+1) % 500 == 0:
            # Save the generated image
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-{}.jpg'.format(step+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train network')
    # general
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--content', type=str, default='./image/content.jpg', help='content image file path')
    parser.add_argument('--style', type=str,default='./image/style.jpg', help='style image file path')
    parser.add_argument('--styleWeight', type=int, default=100, help='weight decay')
    args = parser.parse_args()
    print(args)
    main(args)