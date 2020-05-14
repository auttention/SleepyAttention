import click
import create_dataset
import evaluations
import extract_spectrograms
import network_attention_bi_bi
import network_attention_bi_bi_hidden
import network_seq2seq_bi_bi_fc
import network_seq2seq_bi_bi


@click.command()
@click.argument('command')
@click.option('--output_directory', '-od')
@click.option('--source_directory', '-sd')
@click.option('--output_path', '-op')
@click.option('--filename', '-fn')
@click.option('--checkpoint_directory', '-ckpt_dir')
@click.option('--seq2seq', '-seq')
@click.option('--attention', '-att')
@click.option('--train', '-train')
@click.option('--devel', '-devel')
@click.option('--test', '-test')
@click.option('--predictions', '-pred')
@click.option('--labels', '-lab')
@click.option('--num_feat', '-nfeat')
@click.option('--num_fft', '-nfft')
@click.option('--num_mels', '-nmel')
@click.option('--network_type', '-nt')
@click.option('--batch_size', '-bs')
@click.option('--num_units', '-nu')
@click.option('--num_epochs', '-ne')
@click.option('--learning_rate', '-lr')
def main(command, output_directory, output_path, filename, seq2seq, attention, train, devel, test, predictions,
         labels, num_feat, num_mels, num_fft, network_type, batch_size, num_units, num_epochs, source_directory,
         learning_rate, checkpoint_directory):
    if command == "extract_features":
        if network_type and output_directory and filename and checkpoint_directory:
            if network_type == "seq2seq_bi_bi_fc":
                network_seq2seq_bi_bi_fc.extract_all_features(output_directory, filename, checkpoint_directory=checkpoint_directory)
            if network_type == "seq2seq_bi_bi":
                network_seq2seq_bi_bi.extract_all_features(output_directory, filename, checkpoint_directory=checkpoint_directory)
            if network_type == "attention_bi_bi":
                network_attention_bi_bi.extract_all_features(output_directory, filename, checkpoint_directory=checkpoint_directory)
            if network_type == "attention_bi_bi_hidden":
                network_attention_bi_bi_hidden.extract_all_features(output_directory, filename, checkpoint_directory=checkpoint_directory)

    if command == "fusion_features":
        if seq2seq and attention and output_path:
            evaluations.fusion(seq2seq, attention, output_path)

    if command == "baseline":
        if train and devel and test and predictions and labels and num_feat:
            evaluations.baseline(train, devel, test, predictions, labels, num_feat)

    if command == "test_result":
        if predictions and labels:
            evaluations.eval_sleepiness(predictions, labels)

    if command == "extract_spectrograms":
        if num_mels and num_fft and output_directory and source_directory and labels:
            extract_spectrograms.extract_spectrograms(int(num_fft), int(num_mels), source_directory, output_directory, labels)

    if command == "create_dataset":
        if source_directory:
            create_dataset.create_all_datasets(source_directory)

    if command == "train":
        if network_type and batch_size and num_units and checkpoint_directory and num_epochs and learning_rate:
            if network_type == "seq2seq_bi_bi_fc":
                network_seq2seq_bi_bi_fc.train_network(int(num_epochs), int(batch_size), float(learning_rate), int(num_units), checkpoint_directory)
            if network_type == "seq2seq_bi_bi":
                network_seq2seq_bi_bi.train_network(int(num_epochs), int(batch_size), float(learning_rate), int(num_units), checkpoint_directory)
            if network_type == "attention_bi_bi":
                network_attention_bi_bi.train_network(int(num_epochs), int(batch_size), float(learning_rate), int(num_units), checkpoint_directory)
            if network_type == "attention_bi_bi_hidden":
                network_attention_bi_bi_hidden.train_network(int(num_epochs), int(batch_size), float(learning_rate), int(num_units), checkpoint_directory)


if __name__ == '__main__':
    main()