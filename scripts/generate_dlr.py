""" Generate Deep Learning Resources """
import json
import os
import requests


DOUBLE_NEW_LINE = "\n\n"
USERNAME = os.environ["DLR_USERNAME"]
TOKEN = os.environ["DLR_TOKEN"]
DLR_BACKEND = "https://lnkr-api.zerotosingularity.com/api/dlurl"


def generate_title(tag_for_title):
    """ Generate a title from a tag """
    return tag_for_title.replace("-", " ").title()


def generate_dlr():
    """ Generate the Deep Learning Resources Readme """
    headers = {}
    headers["X-Session-User"] = USERNAME
    headers["X-Session-Token"] = TOKEN
    headers["Content-Type"] = "application/json"

    reply = requests.get(DLR_BACKEND, headers=headers)

    if reply.status_code != requests.codes["ok"]:
        print("DLR generation: Url fetching failed...")
        exit(1)

    tag_list = {}

    urls = json.loads(reply.text)

    for url in urls:
        if url["tags"] is not None:
            matching = [tag for tag in url["tags"] if "dlr" in tag["title"]]

            for match in matching:
                title = match["title"][4:]

                if title not in tag_list:
                    tag_list[title] = []

                tag_list[title].append(url)

    sorted_tag_list = {}

    # Sort tags
    for key in sorted(tag_list):
        sorted_tag_list[key] = tag_list[key]

    # sort urls in tags
    for tag in sorted_tag_list:
        sorted_tag_list[tag] = sorted(
            sorted_tag_list[tag], key=lambda k: k['CreatedAt'])

    dlr_generated = f"# Deep Learning Resources{DOUBLE_NEW_LINE}\
    > Trying to organise the vast majority of\
    Deep Learning resources that I encounter.\
    {DOUBLE_NEW_LINE} If you want to contribute, feel free to make a pull request.\
    {DOUBLE_NEW_LINE} The Readme currently gets generated based on the Lnkr API\
     from [Zero to Singularity](https://zerotosingularity.com) at\
     [https: // lnkr.zerotosingularity.com](https: // lnkr.zerotosingularity.com)\
     which is currently not publicly available yet. Feel free to contact me at\
     jan@zerotosingularity.com if you would like to contribute.\
    {DOUBLE_NEW_LINE}# Table of Contents{DOUBLE_NEW_LINE}"

    count = 1

    for tag in sorted_tag_list:
        title = generate_title(tag)
        dlr_generated += f"{count}. [{title}](#{tag})\n"
        count += 1

    dlr_generated += DOUBLE_NEW_LINE

    for tag in sorted_tag_list:
        title = generate_title(tag)
        dlr_generated += f"## {title}"
        dlr_generated += DOUBLE_NEW_LINE

        count = 1

        for url in sorted_tag_list[tag]:
            is_child = [t for t in url["tags"] if "child" in t["title"]]
            line = ""

            if len(is_child) > 0:
                line = f'  * [{url["title"]}]({url["url"]}]\n'
            else:
                line = f'{count}. [{url["title"]}]({url["url"]})\n'
                count += 1

            dlr_generated += line

        dlr_generated += DOUBLE_NEW_LINE

    new_dlr = open("../README.md", "w")
    new_dlr.write(dlr_generated)


generate_dlr()
