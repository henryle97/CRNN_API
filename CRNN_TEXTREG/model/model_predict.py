from CRNN_TEXTREG.model.model import CRNN
import CRNN_TEXTREG.params as params
import torch
from CRNN_TEXTREG.tool import dataset
from CRNN_TEXTREG import utils
from torch.autograd import Variable
from PIL import Image
import time
import numpy as np
from sklearn.preprocessing import normalize


class TEXTREG(object):
    def __init__(self, path_weights="weights/weights-20.pth"):
        self.model = CRNN(params.height, params.n_channel, len(params.alphabet) + 1, params.number_hidden)
        self.device = torch.device("cpu")
        # self.model.to(torch.device("cuda"))
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(path_weights, map_location="cpu"))
        self.converter = utils.strLabelConverter(params.alphabet)
        self.transformer = dataset.resizeNormalize((128, 32))
        self.model.to(self.device)
        self.model.eval()
        self.dict_text = {}
        for index, char in enumerate(params.alphabet):
            self.dict_text[char] = index

    def predict(self, img):
        '''

        :param img: PIL image
        :return:
        '''
        print(img.size)
        image = self.transformer(img)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.LongTensor([preds.size(0)]))
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return sim_pred

    def predict_batch(self, list_img):
        print(list_img)
        if len(list_img) == 1:
            return self.predict(list_img[0])
        # batch_pred = torch.LongTensor(np.concatenate([self.transformer(img) for img in list_img], axis=0))
        # list_batch = [torch.Tensor(self.transformer(img)) for img in list_img]
        list_batch = [torch.Tensor(self.transformer(img)).to(self.device) for img in list_img]
        batch_pred = torch.cat(list_batch)
        batch_pred = batch_pred.view(len(list_img), 1, batch_pred.size(1), batch_pred.size(2))
        image = Variable(batch_pred)
        # image = image.long()
        preds = self.model(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * len(list_img)))
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
        return " ".join(sim_preds)

if __name__ == "__main__":
    model = TEXTREG(path_weights="/home/hisiter/working/VCCorp/text_mutex/extract_text_from_image/weights/weights-20.pth")
    img = Image.open("/home/hisiter/working/VCCorp/text_mutex/extract_text_from_image/img_test/plate.png").convert('L')
    import time
    t1 = time.time()
    print(model.predict(img))
    print(time.time() - t1)