"""Diagnostics evaluations for BERT."""
import matplotlib.pyplot as plt
import numpy as np
import torch


class Evaluate:

    def __init__(self):
        pass

    def __call__(self, sentence, target_word, k_to_show=10, debug=False,
                 plot=True, display=True):
        self.debug = debug

        input_ids, target_seq_ix = self.get_input_ids(
            sentence, target_word, debug)
        pred_scores = self.get_predictions(input_ids, target_seq_ix)
        n_vocab = pred_scores.shape[0]

        # convert scores to probs; TODO: is this valid?
        pred_probs = torch.softmax(pred_scores, dim=0).detach().cpu().numpy()

        # order the outputs
        ordered_token_ixs = np.argsort(-pred_probs)
        ordered_probs = pred_probs[ordered_token_ixs]

        # determine the vocab index of target word
        target_vocab_ix = self.tokenizer.convert_tokens_to_ids([target_word])[0]

        # determine target word's position in the ordering and prediction prob
        target_order_ix = next(ix for ix in range(n_vocab)
                               if ordered_token_ixs[ix] == target_vocab_ix) + 1
        target_prob = pred_probs[target_vocab_ix]

        if display:
            # print the sentence for reference
            print('-' * 8)
            print(f'"{sentence}"')

            # report target position in distribution
            print('Predicting "%s" @ rank %s / %s (top %3.2f%%) '
                  'with probability %4.3f'
                  % (target_word, target_order_ix, n_vocab,
                     target_order_ix / n_vocab * 100, target_prob))

        if display:
            if plot:
                self.plot_non_zero(ordered_probs, target_order_ix, target_prob)
            self.print_top(k_to_show, ordered_token_ixs, ordered_probs)

        return target_prob, target_order_ix, ordered_probs, ordered_token_ixs

    def get_input_ids(self, sentence, target_word, debug=False):
        # make sure the sentence ends with a period (helps BERT)
        if sentence[-1] != '.':
            print('Sentence does not end with period, adding...')
            sentence += '.'

        # tokenize
        tokens = self.tokenizer.tokenize(sentence)

        # prepend cls and append sep
        tokens = [self.tokenizer.cls_token] + tokens
        tokens.append(self.tokenizer.sep_token)

        # still not sure how to deal with this case
        if target_word not in tokens:
            print(tokens)
            print('Is it in word pieces?')
            print(target_word)
            raise Exception

        # determine the sentence index of the target word
        target_seq_ix = next(ix for ix in range(len(tokens))
                             if tokens[ix] == target_word)

        # replace with mask
        tokens[target_seq_ix] = self.tokenizer.mask_token

        # convert to vocab ixs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        return input_ids, target_seq_ix

    def get_predictions(self, model, target_seq_ix):
        raise NotImplementedError

    @staticmethod
    def plot_non_zero(ordered_probs, target_order_ix, target_prob):
        # plot the non-zero fellas
        m = next(ix for ix in range(len(ordered_probs))
                 if round(ordered_probs[ix], 3) == 0.) - 1
        x = list(range(m))
        y = ordered_probs[0:m]
        plt.plot(x, y)

        # if the target is in this set, plot it's location
        if target_order_ix <= m:
            y = np.linspace(0, max(y))
            x = [target_order_ix] * len(y)
            plt.plot(x, y, 'r--')
            x = list(range(m))
            y = [target_prob] * m
            plt.plot(x, y, 'g--')

        plt.xlabel('Token Rank (non-zero probs, 3 d.p.)')
        plt.ylabel('Prob')
        plt.show()

    def print_top(self, k, ordered_token_ixs, ordered_probs):
        print('Top %s predictions' % k)
        print('\tRank\tToken\t\tProb')
        print('\t----\t-----\t\t----')
        for i in range(k):
            token_ix = int(ordered_token_ixs[i])
            token = self.to_token(token_ix)
            print('\t%s\t%s%s\t%4.3f'
                  % (i + 1,
                     token,
                     '\t' if len(token) < 8 else '',
                     ordered_probs[i]))

    def to_token(self, ix):
        return self.tokenizer.convert_ids_to_tokens([ix])[0]


class EvaluateBert(Evaluate):

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def get_input_ids(self, sentence, target_word, debug=False):
        # make sure the sentence ends with a period (helps BERT)
        if sentence[-1] != '.':
            print('Sentence does not end with period, adding...')
            sentence += '.'

        # tokenize
        tokens = self.tokenizer.tokenize(sentence)

        # still not sure how to deal with this case
        if target_word not in tokens:
            print(tokens)
            print('Is your target word in pieces? If so pick another.')
            print(target_word)
            raise Exception

        # prepend CLS
        tokens = [self.tokenizer.cls_token] + tokens

        # determine the sentence index of the target word
        target_seq_ix = next(ix for ix in range(len(tokens))
                             if tokens[ix] == target_word)

        # mask the token to predict
        tokens[target_seq_ix] = self.tokenizer.mask_token

        # append sep at the end (better results)
        tokens.append(self.tokenizer.sep_token)

        # debug info
        if debug:
            print(f'Target "{target_word}" @ index {target_seq_ix}')
            for ix, token in enumerate(tokens):
                print(f'{ix}\t{token}')

        # recast as sentence
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        return input_ids, target_seq_ix

    def get_predictions(self, input_ids, target_seq_ix):
        # run the model and get the outputs
        outputs = self.model(input_ids, masked_lm_labels=input_ids)
        loss, pred_scores = outputs[:2]
        pred_scores = pred_scores[0][target_seq_ix]
        return pred_scores
