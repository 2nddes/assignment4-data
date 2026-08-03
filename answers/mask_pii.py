from answers.utils import sampling_warc, mask_ip, mask_phone_number, mask_emails


def mask_pii():
    samples = sampling_warc(10)

    for sample in samples:
        result = mask_emails(sample)
        if result[1] != 0:
            print('-' * 20)
            print(sample)
            print('-' * 20)
            print(result[0])
        else:
            print("未替换email")

        sample = result[0]
        result = mask_phone_number(sample)
        if result[1] != 0:
            print('-' * 20)
            print(sample)
            print('-' * 20)
            print(result[0])
        else:
            print("未替换phone")

        sample = result[0]
        result = mask_ip(sample)
        if result[1] != 0:
            print('-' * 20)
            print(sample)
            print('-' * 20)
            print(result[0])
        else:
            print("未替换ip")

if __name__ == "__main__":
    mask_pii()