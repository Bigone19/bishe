def extract_log(log_file, new_log_file, key_word):
    with open(log_file, 'r') as f:
        with open(new_log_file, 'w') as train_log:
            for line in f:
                if 'Syncing' in line:
                    continue
                if 'nan' in line:
                    continue
                if key_word in line:
                    train_log.write(line)
    f.close()
    train_log.close()


extract_log('log.txt', 'training_loss.log', 'images')
extract_log('log.txt', 'training_iou.log', 'IOU')
