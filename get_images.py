import sys
import os

from icrawler.builtin import BingImageCrawler


def usage():
    print(
        f"{sys.argv[0]} <SAVE_FILE_DIR> <SEARCH_WORD> <SAVE_FILE_NUM>", file=sys.stderr)


def main():
    argv = sys.argv

    if len(argv) < 4:
        usage()
        exit(1)

    if not os.path.isdir(argv[1]):
        os.makedirs(argv[1])

    crawler = BingImageCrawler(storage={"root_dir": argv[1]})
    crawler.crawl(keyword=argv[2], max_num=int(argv[3]))


if __name__ == "__main__":
    main()
