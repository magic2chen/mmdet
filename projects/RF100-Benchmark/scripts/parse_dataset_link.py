import re
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(
        description='A handy script that will decompose and print from '
        "a roboflow Data link it's workspace, project and version")
    parser.add_argument(
        '-l', '--link', required=True, help='A link to a roboflow Data')
    args = vars(parser.parse_args())
    # first one gonna be protocol, e.g. http
    _, url, workspace, project, version = re.split('/+', args['link'])
    print(url, workspace, project, version)


if __name__ == '__main__':
    main()
