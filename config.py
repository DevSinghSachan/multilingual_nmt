from argparse import ArgumentParser


def get_train_args():
    parser = ArgumentParser(description='Implementation of Transformer in Pytorch')

    parser.add_argument('--input', '-i', type=str, default='./data/ja_en',
                        help='Input directory')
    parser.add_argument('--data', type=str, default='demo',
                        help='Output file for the prepared data')
    parser.add_argument('--report_every', type=int, default=50,
                        help='Print stats at this interval')
    parser.add_argument('--model', type=str, default='Transformer',
                        help='Model Type to train ( Trasformer / MultiTaskNMT / Shared )')

    # Mulltilingual Options
    parser.add_argument('--pshare_decoder_param', dest='pshare_decoder_param',
                        action='store_true',
                        help='partially share the decoder params for the models')
    parser.set_defaults(pshare_decoder_param=False)
    parser.add_argument('--pshare_encoder_param', dest='pshare_encoder_param',
                        action='store_true',
                        help='partially share the encoder params for the models')
    parser.set_defaults(pshare_encoder_param=False)
    parser.add_argument('--lang1', type=str)
    parser.add_argument('--lang2', type=str)

    parser.add_argument('--share_sublayer', type=str, help='kvqf|+linear')
    parser.add_argument('--attn_share', type=str, help='self|+source')

    # Training Options
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of sentences in each mini-batch')
    parser.add_argument('--wbatchsize', '-wb', type=int, default=4096,
                        help='Number of words in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', dest='resume', action='store_true',
                         help="resume the model training")
    parser.set_defaults(resume=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Compute norm of gradient")
    parser.set_defaults(debug=False)
    parser.add_argument('--grad_accumulator_count', type=int, default=1,
                        help='number of minibatches to accumulate the gradient of')

    # Model Options
    parser.add_argument('--multi_gpu', nargs='+', default=[0], type=int,
                        help='gpu ids')
    parser.add_argument('--n_units', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--layers', '-l', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--multi_heads', type=int, default=8,
                        help='Number of heads in attention mechanism')
    parser.add_argument('--dropout', '-d', type=float, default=0.1)

    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--relu_dropout', type=float, default=0.1)
    parser.add_argument('--layer_prepostprocess_dropout', type=float, default=0.1)

    parser.add_argument('--tied', dest='tied', action='store_true',
                        help='tie target word embedding and output softmax layer')
    parser.set_defaults(tied=False)
    parser.add_argument('--pos_attention', dest='pos_attention', action='store_true',
                        help='positional attention in decoder')
    parser.set_defaults(pos_attention=False)
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Use label smoothing for cross-entropy')
    parser.add_argument('--embed-position', action='store_true',
                        help='Use position embedding rather than sinusoid')
    parser.add_argument('--max_length', type=int, default=500,
                        help='Maximum Possible length for a sequence')
    parser.add_argument('--no_pad_remover', dest='use_pad_remover',
                        action='store_false',
                        help='Use pad remover in FFN Linear layers')
    parser.set_defaults(use_pad_remover=True)

    # Optimizer Options
    parser.add_argument('--optimizer', type=str, default='Noam',
                        help='Optimizer choice (Noam|Adam|Yogi)')
    parser.add_argument('--grad_norm_for_yogi', dest='grad_norm_for_yogi',
                        action='store_true',
                        help='grad norm for Yogi')
    parser.add_argument('--warmup_steps', type=float, default=16000,
                        help='warmup steps in Adam Optimizer Training')
    parser.add_argument('--learning_rate', default=0.2, type=float,
                        help='learning rate')
    parser.add_argument('--learning_rate_constant', default=2.0, type=float,
                        help='learning rate constant')
    parser.add_argument('--optimizer_adam_beta1', default=0.9, type=float,
                        help='Beta1 for Adam training')
    parser.add_argument('--optimizer_adam_beta2', default=0.997, type=float,
                        help='Beta2 for Adam training')
    parser.add_argument('--optimizer_adam_epsilon', default=1e-9, type=float,
                        help='Epsilon for Adam training')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay')

    # Evaluation Options
    parser.add_argument('--eval_steps', default=1000, type=int,
                        help='Number of steps for evaluation')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='Beam size during translation')
    parser.add_argument('--metric', default='bleu', type=str,
                        help='Metric to save the model. Options are: bleu/accuracy')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Length Normalization coefficient')
    parser.add_argument('--max_sent_eval', default=500, type=int,
                        help='Max. sentences to evaluate while training')
    parser.add_argument('--max_decode_len', type=int, default=50)

    # Output Files
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    parser.add_argument('--model_file', default='results/model.ckpt', type=str,
                        help='path to save the model')
    parser.add_argument('--best_model_file', default='results/model_best.ckpt', type=str,
                        help='path to save the best model')
    parser.add_argument('--dev_hyp', default='results/valid.out', type=str,
                        help='path to save dev set hypothesis')
    parser.add_argument('--test_hyp', default='results/test.out', type=str,
                        help='path to save test set hypothesis')
    parser.add_argument('--log_path', default='results/log.txt', type=str,
                        help='logger path')

    args = parser.parse_args()
    return args


def get_preprocess_args():

    """Data Preprocessing Options"""
    parser = ArgumentParser(description='Preprocessing Options')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    parser.add_argument('--tok', dest='tok', action='store_true',
                        help='Vocabulary size of target language')
    parser.set_defaults(tok=False)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--input', '-i', type=str, default='./data/ja_en',
                        help='Input directory')
    parser.add_argument('--source_train', '-s-train', type=str,
                        default='train.ja',
                        help='Filename of train data for source language')
    parser.add_argument('--target_train', '-t-train', type=str,
                        default='train.en',
                        help='Filename of train data for target language')
    parser.add_argument('--source_valid', '-s-valid', type=str,
                        default='dev.ja',
                        help='Filename of validation data for source language')
    parser.add_argument('--target_valid', '-t-valid', type=str,
                        default='dev.en',
                        help='Filename of validation data for target language')
    parser.add_argument('--source_test', '-s-test', type=str,
                        default='test.ja',
                        help='Filename of test data for source language')
    parser.add_argument('--target_test', '-t-test', type=str,
                        default='test.en',
                        help='Filename of test data for target language')
    parser.add_argument('--save_data', type=str, default='demo',
                        help='Output file for the prepared data')
    args = parser.parse_args()
    return args


def get_translate_args():
    parser = ArgumentParser(description='Translate Options')
    parser.add_argument('--input', '-i', type=str, default='./data/ja_en',
                        help='Input directory')
    parser.add_argument('--data', type=str, default='demo',
                        help='Output file for the prepared data')
    parser.add_argument('--src', type=str,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('--tok', dest='tok', action='store_true',
                        help='tokenization option')
    parser.set_defaults(tok=False)
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--output', type=str, default='pred.txt',
                        help='Path to output the predictions (each line will be the decoded sequence')
    parser.add_argument('--model_file', type=str, default='results/model.ckpt',
                        help='Path to model .ckpt file')
    parser.add_argument('--best_model_file', default='results/model_best.ckpt', type=str,
                        help='path to save the best model')
    parser.add_argument('--batchsize', type=int, default=60)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_decode_len', type=int, default=50)
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Length Normalization coefficient')
    parser.add_argument('--model', type=str, default='Transformer',
                        help='Model Type to train ( Trasformer / MultiTaskNMT )')

    args = parser.parse_args()
    return args
