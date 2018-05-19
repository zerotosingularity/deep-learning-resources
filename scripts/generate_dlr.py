import requests
import json
import os


DOUBLE_NEW_LINE = "\n\n"
USERNAME = os.environ["DLR_USERNAME"]
TOKEN = os.environ["DLR_TOKEN"]
DLR_BACKEND = "https://lnkr-api.zerotosingularity.com/api/dlurl"


def generate_title(tag):
    return tag.replace("-", " ").title()


DOUBLE_NEW_Line = "\n\n"

headers = {}
headers["X-Session-User"] = USERNAME
headers["X-Session-Token"] = TOKEN
headers["Content-Type"] = "application/json"

r = requests.get(DLR_BACKEND, headers=headers)

if (r.status_code != requests.codes.ok):
    print("DLR generation: Url fetching failed...")
    exit(1)

tag_list = {}

urls = json.loads(r.text)

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
{DOUBLE_NEW_LINE}# Table of Contents{DOUBLE_NEW_LINE}"

count = 1

for tag in sorted_tag_list:
    title = generate_title(tag)
    toc_line = f"{count}. [{title}](#{tag})\n"
    dlr_generated += toc_line
    count += 1

dlr_generated += DOUBLE_NEW_LINE

for tag in sorted_tag_list:
    title = generate_title(tag)
    dlr_generated += f"## {title}"
    dlr_generated += DOUBLE_NEW_LINE

    count = 1

    for url in sorted_tag_list[tag]:

        isChild = [t for t in url["tags"] if "child" in t["title"]]
        line = ""

        if len(isChild) > 0:
            line = f'  * [{url["title"]}]({url["url"]}]\n'
        else:
            line = f'{count}. [{url["title"]}]({url["url"]})\n'
            count += 1

        dlr_generated += line

    dlr_generated += DOUBLE_NEW_LINE

new_dlr = open("README.md", "w")
new_dlr.write(dlr_generated)
