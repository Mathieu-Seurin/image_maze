def select_fusing():
    pass



def concatenate_text_vision(self, text, vision):
    vision = vision.view(vision.size(0), -1)
    return torch.cat((vision, text), dim=1)


def dot_product_text_vision(self, text, vision):
    vision = vision.view(vision.size(0), -1)

    text = self.text_embedding_before_dot(text)
    vision = self.visual_embedding_before_dot(vision)
    return text * vision


def compute_attention(self, text, vision):
    """
    :param text: lstm-encoded text. dim is (batch, hidden_lstm_size)
    :param vision: cnn-encoded image. dim is (batch, n_feature_map, width, height)
    :return: vision after visual attention is applied. dim is (batch, n_feature_map)
    """
    n_feature_map = vision.size(1)
    width = vision.size(2)
    height = vision.size(3)

    attention_weights_list = []
    # compute attention for every pixel, compute the sum
    for i in range(width):
        for j in range(height):
            current_pixel = vision[:, :, i, j]
            assert current_pixel.dim() == 2
            current_weight = self.attention_last(self.attention_hidden(torch.cat((text, current_pixel), dim=1)))
            attention_weights_list.append(current_weight)

    all_weigths = torch.cat(attention_weights_list, dim=1)
    all_weigths = F.softmax(all_weigths, dim=1).unsqueeze(2)

    vision = vision.view(-1, n_feature_map, height * width)
    vision = torch.bmm(vision, all_weigths)
    vision = vision.squeeze(2)

    return self.concatenate_text_vision(text, vision)


def vectorize(self, text, vision):
    return vision.view(vision.size(0), -1)