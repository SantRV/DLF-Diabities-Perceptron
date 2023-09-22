

from utils.utils import Utils


def main():
    utils = Utils()
    file_name = "diabetes_pre_processed.txt"
    data = utils.load_data(file_name)
    print(data.head())
    return


if __name__ == "__main__":
    main()
