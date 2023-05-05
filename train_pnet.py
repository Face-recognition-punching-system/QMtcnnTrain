import os
from datetime import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchsummary import summary
from torch.utils.data import DataLoader


from loss import ClassLoss, BBoxLoss, LandmarkLoss, accuracy
from pnet import PNet
from data import CustomDataset


if __name__ == '__main__':
    radio_cls_loss = 1.0
    radio_bbox_loss = 0.5
    radio_landmark_loss = 0.5
    data_path = './12/alldata'
    batch_size = 384
    learning_rate = 1e-3
    epoch_num = 30
    model_path = './model'
    device = torch.device("cuda")
    model = PNet()
    model.to(device)
    summary(model, (3, 12, 12))
    train_dataset = CustomDataset(data_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[6, 14, 20], gamma=0.1)
    class_loss = ClassLoss()
    bbox_loss = BBoxLoss()
    landmark_loss = LandmarkLoss()
    for epoch in range(epoch_num):
        for batch_id, (img, label, bbox, landmark) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device).long()
            bbox = bbox.to(device)
            landmark = landmark.to(device)
            class_out, bbox_out, landmark_out = model(img)
            cls_loss = class_loss(class_out, label)
            box_loss = bbox_loss(bbox_out, bbox, label)
            landmarks_loss = landmark_loss(landmark_out, landmark, label)
            total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * box_loss + radio_landmark_loss * landmarks_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if batch_id % 100 == 0:
                acc = accuracy(class_out, label)
                print('[%s] Train epoch %d, batch %d, total_loss: %f, cls_loss: %f, box_loss: %f, landmarks_loss: %f, '
                      'accuracy：%f' % (
                      datetime.now(), epoch, batch_id, total_loss, cls_loss, box_loss, landmarks_loss, acc))
                pass
            pass
        scheduler.step()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            pass
        torch.jit.save(torch.jit.script(model), os.path.join(r'./model/PNet.pth'))
        pass
    pass
