from collections import defaultdict

import numpy as np


class CTCDecoder:
    NINF = -1 * float("inf")
    DEFAULT_EMISSION_THRESHOLD = 0.01

    @staticmethod
    def merge_duplicates_remove_blanks(labels, blank_class=0):
        if len(labels.shape) == 2:
            return [
                CTCDecoder._merge_duplicates_remove_blanks(blank_class, label)
                for label in labels
            ]
        else:
            return CTCDecoder._merge_duplicates_remove_blanks(blank_class, labels)

    @staticmethod
    def _merge_duplicates_remove_blanks(blank_class, labels):
        new_labels = []
        # merge duplicate labels
        previous = None
        for label in labels:
            if label != previous:
                new_labels.append(label)
                previous = label
        # remove blanks
        new_labels = [label for label in new_labels if label != blank_class]
        return new_labels

    @staticmethod
    def greedy_decode(log_prob: np.ndarray, blank_class=0, **kwargs):
        """
        Gets greedy samples
        :param log_prob:
        :param blank_class:
        :return:
        """
        labels = np.argmax(log_prob, axis=-1)
        labels = CTCDecoder.merge_duplicates_remove_blanks(
            labels, blank_class=blank_class
        )
        return labels

    @staticmethod
    def beam_search_decode(emission_log_prob, blank_class=0, **kwargs):
        from scipy.special import (
            logsumexp,
        )  # log(p1 + p2) = logsumexp([log_p1, log_p2])

        beam_size = kwargs["beam_size"]
        emission_threshold = kwargs.get(
            "emission_threshold", np.log(CTCDecoder.DEFAULT_EMISSION_THRESHOLD)
        )

        length, class_count = emission_log_prob.shape

        beams = [([], 0)]  # (prefix, accumulated_log_prob)
        for t in range(length):
            new_beams = []
            for prefix, accumulated_log_prob in beams:
                for c in range(class_count):
                    log_prob = emission_log_prob[t, c]
                    if log_prob < emission_threshold:
                        continue
                    new_prefix = prefix + [c]
                    # log(p1 * p2) = log_p1 + log_p2
                    new_accu_log_prob = accumulated_log_prob + log_prob
                    new_beams.append((new_prefix, new_accu_log_prob))

            # sorted by accumulated_log_prob
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        # sum up beams to produce labels
        total_accu_log_prob = {}
        for prefix, accu_log_prob in beams:
            labels = tuple(
                CTCDecoder.merge_duplicates_remove_blanks(
                    prefix, blank_class=blank_class
                )
            )
            # log(p1 + p2) = logsumexp([log_p1, log_p2])
            total_accu_log_prob[labels] = logsumexp(
                [accu_log_prob, total_accu_log_prob.get(
                    labels, CTCDecoder.NINF)]
            )

        labels_beams = [
            (list(labels), accu_log_prob)
            for labels, accu_log_prob in total_accu_log_prob.items()
        ]
        labels_beams.sort(key=lambda x: x[1], reverse=True)
        labels = labels_beams[0][0]

        return labels

    @staticmethod
    def prefix_beam_decode(emission_log_prob, blank=0, **kwargs):
        from scipy.special import (
            logsumexp,
        )  # log(p1 + p2) = logsumexp([log_p1, log_p2])

        beam_size = kwargs["beam_size"]
        emission_threshold = kwargs.get(
            "emission_threshold", np.log(CTCDecoder.DEFAULT_EMISSION_THRESHOLD)
        )

        length, class_count = emission_log_prob.shape

        beams = [
            (tuple(), (0, CTCDecoder.NINF))
        ]  # (prefix, (blank_log_prob, non_blank_log_prob))
        # initial of beams: (empty_str, (log(1.0), log(0.0)))

        for t in range(length):
            new_beams_dict = defaultdict(
                lambda: (CTCDecoder.NINF, CTCDecoder.NINF)
            )  # log(0.0) = NINF

            for prefix, (lp_b, lp_nb) in beams:
                for c in range(class_count):
                    log_prob = emission_log_prob[t, c]
                    if log_prob < emission_threshold:
                        continue

                    end_t = prefix[-1] if prefix else None

                    # if new_prefix == prefix
                    new_lp_b, new_lp_nb = new_beams_dict[prefix]

                    if c == blank:
                        new_beams_dict[prefix] = (
                            logsumexp(
                                [new_lp_b, lp_b + log_prob, lp_nb + log_prob]),
                            new_lp_nb,
                        )
                        continue
                    if c == end_t:
                        new_beams_dict[prefix] = (
                            new_lp_b,
                            logsumexp([new_lp_nb, lp_nb + log_prob]),
                        )

                    # if new_prefix == prefix + (c,)
                    new_prefix = prefix + (c,)
                    new_lp_b, new_lp_nb = new_beams_dict[new_prefix]

                    if c != end_t:
                        new_beams_dict[new_prefix] = (
                            new_lp_b,
                            logsumexp(
                                [new_lp_nb, lp_b + log_prob, lp_nb + log_prob]),
                        )
                    else:
                        new_beams_dict[new_prefix] = (
                            new_lp_b,
                            logsumexp([new_lp_nb, lp_b + log_prob]),
                        )

            # sorted by log(blank_prob + non_blank_prob)
            beams = sorted(
                new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True
            )
            beams = beams[:beam_size]

        labels = list(beams[0][0])
        return labels

    @staticmethod
    def ctc_decode_batch(
        log_probs: np.ndarray,
        label2char=None,
        blank_class=0,
        decoder_name="greedy",
        beam_size=10,
    ):
        """
        Input array should be [batch, length, class]
        :param log_probs:
        :param label2char:
        :param blank_class:
        :param decoder_name:
        :param beam_size:
        :return:
        """

        decoded_list = [
            CTCDecoder.ctc_decode(
                log_prob, decoder_name, blank_class, label2char, beam_size
            )
            for log_prob in log_probs
        ]
        return decoded_list

    @staticmethod
    def ctc_decode(
        log_prob: np.ndarray,
        decoder_name="greedy",
        blank_class=0,
        label2char=None,
        beam_size=10,
    ):
        decoder = {
            "greedy": CTCDecoder.greedy_decode,
            "beam_search": CTCDecoder.beam_search_decode,
            "prefix_beam_search": CTCDecoder.prefix_beam_decode,
        }[decoder_name]
        decoded = decoder(log_prob, blank_class=blank_class,
                          beam_size=beam_size)
        if label2char:
            decoded = CTCDecoder.single_decode(decoded, label2char)
        return decoded

    @staticmethod
    def single_decode(prediction: list, label2char: dict):
        return [label2char[p] for p in prediction]

    @staticmethod
    def decode(predictions: list, label2char: dict):
        if isinstance(predictions[0], list):
            return [
                CTCDecoder.single_decode(prediction, label2char)
                for prediction in predictions
            ]
        else:
            return CTCDecoder.single_decode(predictions, label2char)
