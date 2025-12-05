from torchvision import transforms

INPUT_SIZE = 224
PRE_MEAN = [0.485, 0.456, 0.406]
PRE_STD = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=PRE_MEAN, std=PRE_STD)
])
