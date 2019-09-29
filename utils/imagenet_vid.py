import collections
import numpy as np

def load_imagenet_vid_map(imagenet_index_map, imagenet_to_vid_map):
    """
    Args:
        imagenet_index_map (dict): Maps imagenet index to tuple of
            (wordnet id, label name)
        imagenet_to_vid_map (dict): Maps ImageNet wordnet ids to closest
            ImageNet Vid wordnet ids, as output by generate_wnid_map.py.

    Returns:
        vid_index_map (dict): Map imagenet label indices to tuple of (ImageNet
            Vid wordnet id, label name). If the index does not correspond to
            any label from ImageNet Vid, it is omitted.
    """
    output = {}
    for index, (wordnet_id, label_name) in imagenet_index_map.items():
        if wordnet_id in imagenet_to_vid_map:
            vid_wordnet_id = imagenet_to_vid_map[wordnet_id]
            output[index] = (vid_wordnet_id, label_name)
    return output


def convert_predictions(predictions,
                        imagenet_index_map,
                        imagenet_vid_index_map,
                        imagenet_to_vid_map,
                        aggregation='max'):
    """
    Args:
        predictions (np.array): Shape (num_frames, num_labels), where
            num_labels=1000.
        imagenet_index_map (dict): Maps imagenet index to tuple of
            (wordnet id, label name)
        imagenet_vid_index_map (dict): As with imagenet_index_map, but for
            ImageNet Vid labels.
        imagenet_to_vid_map (dict): Maps ImageNet wordnet ids to closest
            ImageNet Vid wordnet ids, as output by generate_wnid_map.py.
        aggregation (str): Either 'max' or 'avg'. Determines how to aggregate
            predictions for different imagenet classes that map to the same
            imagenet vid class.

    Returns:
        vid_predictions (np.array): Shape (num_frames, num_vid_labels),
            where num_vid_labels=30.
    """
    imagenet_index_to_vid = load_imagenet_vid_map(imagenet_index_map,
                                                  imagenet_to_vid_map)
    # Map imagenet vid wordnet id to list of imagenet label indices
    vid_wordnet_indices = collections.defaultdict(list)
    for index, (wordnet_id, _) in imagenet_index_to_vid.items():
        vid_wordnet_indices[wordnet_id].append(int(index))

    num_frames = predictions.shape[0]
    vid_predictions = np.zeros((num_frames, len(imagenet_vid_index_map)))
    for i, (wordnet_id, _) in imagenet_vid_index_map.items():
        if wordnet_id == 'n00001740':
            # Ignore the background class
            break
        i = int(i)
        mapped_labels = predictions[:, vid_wordnet_indices[wordnet_id]]
        if aggregation == 'max':
            vid_predictions[:, i] = mapped_labels.max(axis=1)
        elif aggregation == 'avg':
            vid_predictions[:, i] = mapped_labels.mean(axis=1)
        else:
            raise ValueError('Unknown aggregation: %s' % aggregation)
    return vid_predictions
