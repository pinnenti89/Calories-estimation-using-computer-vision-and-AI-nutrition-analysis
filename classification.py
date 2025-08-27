import os
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 148)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class_names = ['achar', 'aloo gobi', 'aloo matar', 'aloo methi', 'aloo puri', 'aloo tikki', 'appam', 'apple', 'apple pie',
               'bagels', 'baingan bharta', 'banana', 'basundi', 'beetroot', 'besan cheela', 'besan laddu', 'bhel puri',
               'bhindi masala', 'biryani', 'boondi', 'brownie', 'butter chicken', 'cabbage', 'cake', 'canned potatoes',
               'capsicum', 'carrots', 'cauliflower', 'chai', 'chana masala', 'chapati', 'chicken rezala',
               'chicken tikka', 'chicken tikka masala', 'chilli pepper', 'chilli potato', 'chole bhature', 'chop suey',
               'chow mein', 'cooked oatmeal', 'cooked pasta', 'corn', 'cucumber', 'dal makhani', 'dal tadka', 'dhokla',
               'doughnut', 'dum aloo', 'fried chicken', 'fried fish', 'fried rice', 'gajar ka halwa', 'garlic', 'ginger',
               'gobi manchurian', 'grape', 'gujiya', 'gulab jamun', 'hot dogs', 'ice cream', 'idli', 'imarti', 'jalebi',
               'kachori', 'kadai paneer', 'kadhi pakoda', 'kaju katli', 'kalakand', 'kathi roll', 'kebabs', 'khandvi',
               'khichdi', 'khubani ka meetha', 'kiwi', 'kofta', 'kulfi', 'lassi', 'lemon', 'lemonade', 'lettuce',
               'litti chokha', 'macaroni salad', 'malpua', 'mango', 'masala dosa', 'medu vada', 'mishti doi', 'missi roti',
               'modak', 'momos', 'muffin', 'mysore pak', 'naan bread', 'navratan korma', 'omelette', 'onion',
               'onion pakoda', 'orange', 'palak paneer', 'pan cake', 'pan-fried prawns', 'paneer butter masala',
               'pani puri', 'papad', 'paprika', 'paratha', 'pav bhaji', 'peanut chikki', 'pear', 'peas', 'peda',
               'phirni', 'pineapple', 'pizza', 'poha', 'pomegranate', 'popcorn', 'rabri', 'radish', 'raj kachori',
               'rajma', 'ras malai', 'rasgulla', 'rice cooked', 'samosa', 'sandwich', 'scrambled eggs', 'shankarpali',
               'sheer khurma', 'sheera', 'shelled soy bean', 'shrikhand', 'spinach', 'spring rolls', 'sprouts',
               'stuffed karela', 'sunny side up eggs', 'sweet potatoes', 'taco', 'tiramisu', 'toast', 'tomato',
               'turnip', 'uttapam', 'vada pav', 'veggie burger', 'waffle', 'watermelon']

def classify_images(model, transform, segment_folder, root_folder="project"):
    masks_folder = os.path.join(root_folder, "masks")
    os.makedirs(masks_folder, exist_ok=True)
    
    # Clear existing files in masks folder
    for file in os.listdir(masks_folder):
        os.remove(os.path.join(masks_folder, file))

    predictions = []
    for img_name in os.listdir(segment_folder):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(segment_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                label = class_names[predicted.item()]
            
            new_filename = f"{label}.png"
            if "part0" in img_name:
                new_path = os.path.join(root_folder, new_filename)
            else:
                new_path = os.path.join(masks_folder, new_filename)

            img.save(new_path)
            predictions.append((img_name, label, new_path))
    
    return predictions