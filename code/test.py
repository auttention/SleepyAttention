import extract_spectrograms
import network_attention_bi_bi_hidden

"""
def evaluate_score(nfft, mels):
    extract_spectrograms.extract_spectrograms(nfft, mels)
    shutil.rmtree('/spectrograms', ignore_errors=True)
    shutil.rmtree('/featrures', ignore_errors=True)
    shutil.rmtree('/prediction', ignore_errors=True)

    network_seq2seq.extract_all_features("features/seq2seq/", "seq2seq_features")
    network_attention.extract_all_features("features/attention/", "attention_features")
    fusion.fusion("features/seq2seq/seq2seq_features.train.csv", "features/attention/attention_features.train.csv ", "features/fusion/fusion.train.csv")
    fusion.fusion("features/seq2seq/seq2seq_features.test.csv", "features/attention/attention_features.test.csv ", "features/fusion/fusion.test.csv")
    fusion.fusion("features/seq2seq/seq2seq_features.devel.csv", "features/attention/attention_features.devel.csv ", "features/fusion/fusion.devel.csv")
"""


if __name__ == "__main__":

    #network_attention_bi_bi_hidden.extract_all_features("attention_bi_bi_hidden_features", "features")
    extract_spectrograms.extract_spectrograms(2048, 240, "D:/ComParE2019_ContinuousSleepiness/wav", "spectrograms_2/", "labels/labels.csv")
    #extract_spectrograms.normalize_spectrograms()
    #extract_spectrograms.padding_spectrograms()
    #network_seq2seq.extract_all_features("features/seq2seq/", "seq2seq_features")

    #extract_spectrograms.extract_spectrograms(3600, 240, "spectrograms_3600/")
    #extract_spectrograms.extract_spectrograms(3700, 240, "spectrograms_3700/")
    #extract_spectrograms.extract_spectrograms(3800, 240, "spectrograms_3800/")

"""
    while True:
        temp_score = evaluate_score(max["nfft"] * 2, n_mels)
        if temp_score < max["nfft"]:
            best_score = binary_search(max["nfft"], max["nfft"] * 2)
            print("best test score:", max)
            break
        max = {"nfft": max["nfft"] * 2, "value": temp_score}
"""


