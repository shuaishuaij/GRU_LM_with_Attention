# coding=utf-8
from omnibox.Torch import *
from omnibox.tools import *

import torch
from torch.jit import script, trace
import torch.nn as nn
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

from configs import *

# PARAMERTER ENVIRONMENT &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# META SETTING
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 0  # Minimum word count threshold for trimming

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

DECODE_MAX_LENGTH = 5


# MODEL SETTING
class ModelConfig1(object):
    model_name = 'Yidhra_debug1'
    decoder_attn_method = 'dot'  # 'dot','general','concat'
    hidden_size = 500
    teacher_forcing_ratio = 1.0
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64


# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000


# TRAINING/OPTIMIZATION SETTING
class ExpConfig1(object):
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.001
    decoder_learning_ratio = 5.0
    n_iteration = 20
    print_every = 1
    save_every = 5
    save_dir = os.path.join("data", "save")


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# UTILS &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = - \
        torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# DATA PREPRO &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
class Voc:
    """dictionary class
    """

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words),
                                                   len(self.word2index),
                                                   len(keep_words) / len(
                                                       self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


def readVocs(datafile, corpus_name):
    """load in raw data into Voc dictionary, and return a empty voc and
    data file
    """
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8'). \
        read().strip().split('\n')
    # Split every line into pairs and normalize
    voc = Voc(corpus_name)
    for i in range(len(lines)):
        line = lines[i]
        voc.addSentence(line)
    return voc, lines


def trimRareWords(voc, lines, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_lines = []
    for line in lines:
        keep = True
        # Check input sentence
        for word in line.split(' '):
            if word not in voc.word2index:
                keep = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or
        # output sentence
        if keep:
            keep_lines.append(line)

    print("Trimmed from {} lines to {}, {:.4f} of total".format(
        len(lines), len(keep_lines), len(keep_lines) / len(lines)))
    return keep_lines


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in
            sentence.split(' ')] + [
               EOS_token]


def zeroPadding(length, fillvalue=PAD_token):
    """快速padding
    """
    return list(itertools.zip_longest(*length, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    """l is the input batch
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, lines_batch):
    lines_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch = []
    for line in lines_batch:
        input_batch.append(line)
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(input_batch, voc)
    return inp, lengths, output, mask, max_target_len


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# MODEL &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

class BiGruEncoder(nn.Module):
    def __init__(self, config, embedding, vocab_size, gru_dropout=0.1):
        super(BiGruEncoder, self).__init__()
        self.hidden_dim = config.hidden_size
        self.vocab_size = vocab_size
        self.n_layers = config.encoder_n_layers

        # layers
        self.embedding_layer = embedding
        self.gru_layer = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            self.n_layers,
            dropout=(
                0 if self.n_layers == 1 else gru_dropout),
            bidirectional=True)

    def forward(self, input, input_length, hidden_state=None):
        embed_seq = self.embedding_layer(input)
        packed_embed_seq = nn.utils.rnn.pack_padded_sequence(
            embed_seq, input_length)
        output, hidden = self.gru_layer(packed_embed_seq, hidden_state)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # Concat bidirectional GRU outputs
        output = output[:, :, :self.hidden_dim] + \
                 output[:, :, self.hidden_dim:]

        return output, hidden


# attention layers

class Attention(nn.Module):
    def __init__(self, method, hidden_dim):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             "is not an appropriate attention method.")
        self.hidden_dim = hidden_dim
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_dim))

    def dot_score(self, hidden_state, encoder_output):
        return torch.sum(hidden_state * encoder_output, dim=2)

    def general_score(self, hidden_state, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden_state * energy, dim=2)

    def concat_score(self, hidden_state, encoder_output):
        energy = self.attn(
            torch.cat(
                (hidden_state.expand(
                    encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden_state, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden_state, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden_state, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden_state, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added
        # dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# Decoder

class GlobalAttnDecoder(nn.Module):
    def __init__(self,
                 config,
                 embedding,
                 vocab_size,
                 dropout_rate=0.1):
        super(GlobalAttnDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = config.hidden_size
        self.n_layers = config.decoder_n_layers
        self.dropout_rate = dropout_rate
        self.attention_method = config.decoder_attn_method

        # layers
        self.embedding_layer = embedding
        self.embed_dropout = nn.Dropout(dropout_rate)
        self.gru_layer = nn.GRU(
            self.hidden_dim, self.hidden_dim, self.n_layers, dropout=(
                0 if self.n_layers == 1 else dropout_rate))
        self.concat_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)
        self.attn_layer = Attention(self.attention_method, self.hidden_dim)

    def forward(self, input_step, last_hidden, encoder_output):
        # Get embedding of current input word
        teacher = self.embedding_layer(input_step)
        teacher = self.embed_dropout(teacher)
        # Forward through uni-directional GRU
        rnn_output, hidden = self.gru_layer(teacher, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_w = self.attn_layer(rnn_output, encoder_output)
        # Multiply attention weights to encoder outputs to get new "weighted
        # sum" context vector NOTE: USE BMM(BATCH MATRIX MULTIPLICATION)
        contextual_embed = attn_w.bmm(encoder_output.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        contextual_embed = contextual_embed.squeeze(1)
        # SIMPLE CONCAT --> [RNN_OUTPUT ; CONTEXTUAL_EMBED]
        concat_input = torch.cat((rnn_output, contextual_embed), 1)
        # USE A FC LAYER TO MAP self.hidden_dim * 2 --> self.hidden_dim AND
        # ACTIVATION
        concat_output = torch.tanh(self.concat_layer(concat_input))
        # Predict next word
        # A FC MAP hidden_dim --> vocab_size
        output = self.output_layer(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# TRAIN &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def train(
        config,
        input_variable,
        lengths,
        target_variable,
        mask,
        max_target_len,
        encoder,
        decoder,
        embedding,
        encoder_optimizer,
        decoder_optimizer,
        batch_size,
        clip):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < \
                                  config.teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(
                mask_loss.item() *
                nTotal)  # total loss = n*mean loss
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(
        model_config,
        model_name,
        voc,
        pairs,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        embedding,
        encoder_n_layers,
        decoder_n_layers,
        save_dir,
        n_iteration,
        batch_size,
        print_every,
        save_every,
        clip,
        corpus_name,
        loadFilename):
    # Load batches for each iteration
    training_batches = [
        batch2TrainData(
            voc, [
                random.choice(pairs) for _ in range(batch_size)]) for _ in
        range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]

        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(
            input_variable,
            lengths,
            target_variable,
            mask,
            max_target_len,
            encoder,
            decoder,
            embedding,
            encoder_optimizer,
            decoder_optimizer,
            batch_size,
            clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir,
                                     model_name,
                                     corpus_name,
                                     '{}-{}_{}'.format(encoder_n_layers,
                                                       decoder_n_layers,
                                                       model_conf.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory,
                            '{}_{}.tar'.format(iteration, 'checkpoint')))


def trainIters(
        model_conf,
        model_name,
        voc,
        lines,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        embedding,
        encoder_n_layers,
        decoder_n_layers,
        save_dir,
        n_iteration,
        batch_size,
        print_every,
        save_every,
        clip,
        corpus_name,
        loadFilename):
    # Load batches for each iteration
    training_batches = [
        batch2TrainData(
            voc, [
                random.choice(lines) for _ in range(batch_size)]) for _ in
        range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]

        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(
            model_conf,
            input_variable,
            lengths,
            target_variable,
            mask,
            max_target_len,
            encoder,
            decoder,
            embedding,
            encoder_optimizer,
            decoder_optimizer,
            batch_size,
            clip)
        print_loss += loss

        # evaluate every epoch
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(model_conf, exp_conf, encoder, decoder)
        evaluate_epoch(model_conf, exp_conf, encoder, decoder, lines, searcher,
                       voc)
        encoder.train()
        decoder.train()

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir,
                                     model_name,
                                     corpus_name,
                                     '{}-{}_{}'.format(encoder_n_layers,
                                                       decoder_n_layers,
                                                       model_conf.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory,
                            '{}_{}.tar'.format(iteration, 'checkpoint')))


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# TEST &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def mask_decoder_output(decoder_output, context, voc):
    # create masks
    indexes_batch = indexesFromSentence(voc, context)
    mask1 = torch.tensor([1e-6]).expand_as(decoder_output[0]).numpy().tolist()

    for idx in indexes_batch:
        mask1[idx] = 1.
    mask1 = torch.from_numpy(np.array(mask1))

    # masking

    masked_decoder_output1 = decoder_output[0].mul(mask1.type_as(
        decoder_output[0]))

    masked_decoder_output1 = masked_decoder_output1.expand_as(decoder_output)

    return masked_decoder_output1


class GreedySearchDecoder(nn.Module):
    def __init__(self, model_conf, exp_conf, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length,
                context=None, voc=None):
        # Forward input through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the
        # decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token

        # &&&&&&&&&&&&&&&&&& NOTICE &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # ALERT: START FROM SOS THEN MODEL WILL PREDICT FROM BEGINNING
        # WE WANT MODEL TO COMPLETE OUR SENTENCE, THEN WE SHOULD ORDER
        # IT TO START FROM THE LAST WORD OF INPUT SENTENCE:
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        decoder_input = torch.ones(
            1, 1, device=device, dtype=torch.long) * SOS_token

        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            if context:
                decoder_output = mask_decoder_output(decoder_output,
                                                     context, voc)

            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, context,
             max_length=DECODE_MAX_LENGTH):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length,
                              context, voc)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Normalize sentence
            # input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(
                encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [
                x for x in output_words if not (
                        x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def computeF1_macro(pred, golden):
    pred_len = len(pred)
    golden_len = len(golden)
    overlap = list()
    if pred_len <= golden_len:
        for i in range(pred_len):
            if pred[i] in golden:
                overlap.append(pred[i])
    else:
        for i in range(golden_len):
            if golden[i] in pred:
                overlap.append(golden[i])

    precision = (float(len(overlap)) + 1e-10) / golden_len
    recall = (float(len(overlap)) + 1e-10) / pred_len
    F1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, F1


def evaluate_epoch(model_conf, exp_conf, encoder, decoder, data, searcher,
                   voc, fixed_test=True):
    if not fixed_test:
        # select random sequence
        data_size = len(data)
        random_line_id = random.choice(range(data_size))
        sequence = lines[random_line_id]
        seq_word_lst = sequence.strip().split(' ')
        input_sentence = ' '.join(seq_word_lst[:-DECODE_MAX_LENGTH])
        golden = ' '.join(seq_word_lst[-DECODE_MAX_LENGTH:])
    else:
        sequence = "苏 小姐 一向 瞧不起 这 个 寒 碜 的 孙太太 ， 而且 最 不 喜欢 小孩子 ， 可是 听 了 这些 话 ， 心上 高兴 ， 倒 和气 地 笑 道 ：“ 让 他来 ， 我 最 喜欢 小孩子 。” 她 脱 下 太阳 眼镜 ， 合 上 对 着 出神 的 书 ， 小心翼翼 地 握住 了 孩子 的 手腕 ， 免得 在 自己 衣服 上 乱 擦 ， 问 他 道 ：“ 爸爸 呢 ？” 小孩子 不 回答 ， 睁 大 了 眼 ， 向 苏 小姐 “ 波 ！ 波 ！” 吹 唾沫 ， 学餐室 里养 的 金鱼 吹气 泡 。 苏 小姐 慌 得 松 了 手 ， 掏 出手 帕 来自 卫 。 母亲 忙 使劲 拉 他 ， 嚷 着 要 打 他 嘴巴 ， 一 面 叹气 道 ：“ 他 爸爸 在 下面 赌钱 ， 还 用 说么 ！ 我 不 懂 为什么 男人 全 爱 赌 ， 你 看 咱们 同船 的 几 位 ， 没 一个 不 赌得 错天 黑 地 。 赢 几 个 钱 回来 ， 还 说 得 过 。 像 我们 孙 先生 输 了 不少 钱 ， 还要 赌 ， 恨 死 我 了 ！” 苏 小姐 听 了 最后 几 句 小家子气 的 话 ， 不由 心里 又 对 孙 太 太 鄙 夷 ， 冷冷 说道 ：“ 方 先生 倒 不 赌 。” 孙太太 鼻 孔朝天 ， 出冷气 道 ：“ 方 先生 ！ 他 下船 的 时候 也 打 过牌 。 现在 他 忙 着 追求 鲍 小姐 ， 当然 分 不 出 工夫 来 。 人家 终身 大事 ， 比 赌钱 要 紧 得 多 呢 。 我 就 看 不 出 鲍 小姐 又 黑 又 粗 ， 有 什么 美 ， 会 引得 方 先生 好好 二 等 客人 不 做 ， 换 到 三等舱 来 受 罪 。 我 看 他们 俩 要 好 得 很 ， 也许 到 香港 ， 就 会 订婚 。 这 真是 ‘ 有 缘 千 里 来 相会 ’ 了 。”"
        input_sentence = '我 不 懂'
        golden = "为什么 男人 全 爱 赌"

        # Evaluate sentence
    output_words = evaluate(
        encoder, decoder, searcher, voc, input_sentence, sequence)
    # Format and print response sentence
    output_words[:] = [
        x for x in output_words if not (
                x == 'EOS' or x == 'PAD')]

    print('whole sentance: ', sequence)
    print('golden: ', golden)
    print('pred:', ' '.join(output_words))

    # indices
    precision, recall, F1 = computeF1_macro(' '.join(output_words),
                                            sequence[-DECODE_MAX_LENGTH:])
    print("P: ", precision, "R: ", recall, 'F1: ', F1)

    with codecs.open('{}_log.txt'.format(model_conf.model_name), 'a',
                     'utf-8') as f:
        f.write('whole sentance: ' + sequence + '\n')
        f.write('golden: ' + golden + '\n')
        f.write('pred:' + ' '.join(output_words) + '\n')
        f.write("P: " + str(precision) + " R: " + str(recall) + ' F1: ' +
                str(F1) + '\n')


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# RUN &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
if __name__ == '__main__':
    data_file_name = "weicheng_words_2.txt"
    data_file_path = os.path.join("./data", data_file_name)
    mode = 'train'  # train/text

    # configure settings
    model_conf = ModelConfig2
    exp_conf = ExpConfig2

    voc, lines = readVocs(data_file_path, data_file_name)

    if mode == 'test':
        # Set checkpoint to load from; set to None if starting from scratch
        loadFilename = None
        checkpoint_iter = 4000
        # loadFilename = os.path.join(save_dir, model_name, corpus_name,
        #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
        #                            '{}_checkpoint.tar'.format(checkpoint_iter))

        # Load model if a loadFilename is provided
        if loadFilename:
            # If loading on same machine the model was trained on
            checkpoint = torch.load(loadFilename)
            # If loading a model trained on GPU to CPU
            # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            voc.__dict__ = checkpoint['voc_dict']

    # data preprocessing:
    # Trim voc and lines

    lines = trimRareWords(voc, lines, MIN_COUNT)

    # # Example for validation
    # small_batch_size = 5
    #
    # batches = batch2TrainData(voc, [random.choice(lines)
    #                                 for _ in range(small_batch_size)])
    # input_variable, lengths, target_variable, mask, max_target_len = batches
    #
    # print("input_variable:", input_variable)
    # print("lengths:", lengths)
    # print("target_variable:", target_variable)
    # print("mask:", mask)
    # print("max_target_len:", max_target_len)

    # model construction
    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, model_conf.hidden_size)

    loadFilename = False
    if loadFilename:
        embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = BiGruEncoder(model_conf, embedding,
                           voc.num_words,
                           model_conf.dropout)

    decoder = GlobalAttnDecoder(
        model_conf,
        embedding,
        voc.num_words,
        model_conf.dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # experiment setting initialization
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(),
                                   lr=exp_conf.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(),
                                   lr=exp_conf.learning_rate * exp_conf.decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    trainIters(
        model_conf,
        model_conf.model_name,
        voc,
        lines,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        embedding,
        model_conf.encoder_n_layers,
        model_conf.decoder_n_layers,
        exp_conf.save_dir,
        exp_conf.n_iteration,
        model_conf.batch_size,
        exp_conf.print_every,
        exp_conf.save_every,
        exp_conf.clip,
        data_file_name,
        loadFilename)
    # # Set dropout layers to eval mode
    # encoder.eval()
    # decoder.eval()
    #
    # # Initialize search module
    # searcher = GreedySearchDecoder(encoder, decoder)
    #
    # # Begin chatting (uncomment and run the following line to begin)
    # # evaluateInput(encoder, decoder, searcher, voc)

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
