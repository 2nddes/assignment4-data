from answers.utils import sampling_warc, identify_language


def show_rand_lang_identify():
    samples = sampling_warc(20)
    for sample in samples:
        results = identify_language(sample)
        print(results)
        print(sample[:30])

