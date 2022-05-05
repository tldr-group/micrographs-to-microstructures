import requests  # to get image from the web
from bs4 import BeautifulSoup
import re

micro_nums = ['001', '002', '006', '007', '010', '011', '016', '017', '021',
              '031', '039', '043', '047', '048', '051', '053', '054', '060',
              '066', '068', '070', '072', '084', '085', '160', '161', '177',
              '186', '188', '189', '205', '209', '210', '211', '213', '217',
              '227', '228', '233', '235', '237', '238', '276', '277', '286',
              '319', '329', '340', '342', '360', '365', '368', '370', '372',
              '376', '378', '381', '387', '393', '396', '406', '420', '429',
              '441', '442', '447', '477', '484', '516', '545', '612', '630',
              '631', '638', '711', '716', '721', '722', '725', '736', '737',
              '740', '760', '782', '784', '797', '860']

text_to_key = {
    "keyword": ["Keywords", "list"],
    "category": ["Categories", "list"],
    "brief_description": ["Brief description", "str"],
    "long_description": ["Further information", "str"],
    "element": ["Composition", "list"],
    "technique": ["Technique", "str"],
    "contributor": ["Contributor", "str"],
    "organisation": ["Organisation", "str"]
  }

micro_dict = dict()


def return_comma_list(text_from_site):
    """
    :param text_from_site: The scraped text that is divided by commas
    :return: A list of the different texts that are divided by commas,
    without unnecessary spaces.
    """
    if text_from_site:
        txt_list = text_from_site.split(",")
        txt_list = [txt.replace(u'\xa0', u'') for txt in txt_list]
        txt_list = [re.findall(r"\w.*", txt) for txt in
                    txt_list]
        txt_list = [txt[0] if txt else "" for txt in txt_list]
        return txt_list


def find_and_print_text(soup_block, header):
    """
    :param soup_block: The beautifulsoup object
    :param header: The header for the text to be returned
    :return: (Boolean, text) where the Boolean is true iff the header exists
    in the soup block, and the appropriate text if it exist.
    """
    header_txt, list_or_str = header
    if soup_block.find(text=header_txt):
        if soup_block.findNext().find(text=header_txt):
            text_from_site = soup_block.findNext('dd').text
            if list_or_str == "str":
                return text_from_site
            else:  # it needs to be a list
                if header_txt in ["Keywords", "Categories"]:
                    return return_comma_list(text_from_site)
                if header_txt == "Composition":
                    comma_list = return_comma_list(text_from_site)
                    element_list = [re.findall(r"[a-zA-Z]*", txt)[0] for txt in
                                    comma_list]
                    bool1_list = [bool(re.findall(r"^[a-zA-Z]+\s\d", txt)) for
                                 txt in comma_list]
                    bool2_list = [bool(re.findall(r"^[a-zA-Z]+$", txt)) for
                                 txt in comma_list]
                    return [element_list[i] if bool1_list[i] or bool2_list[i]
                            else "long description" for i in
                            range(len(comma_list))]

            # print(f"{header}: {soup_block.findNext('dd').text}")


for micro_num in micro_nums:
    # Get the micrograph number:
    num_wo_0 = re.compile(r"[1-9]\d*").findall(micro_num)[0]

    inside_dict = {"name": f"Micrograph {num_wo_0}"}
    # Get the url:
    record_url = f'https://www.doitpoms.ac.uk/miclib/full_record.php?id' \
                 f'={num_wo_0}'
    record_request = requests.get(record_url)
    soup = BeautifulSoup(record_request.content, 'html.parser')
    # Where the relevant information is:
    soup_info = soup.find('div', class_='col-md-8')
    if soup_info:  # if the page exists:
        soup_info = soup_info.findAll()
        print(f'Micrograph number: {num_wo_0}')
        for block in soup_info:
            for key, value in text_to_key.items():
                inside_dict_value = find_and_print_text(block, value)
                if inside_dict_value:
                    if key in ["keyword", "category", "element"]:
                        print(inside_dict_value)

    print()
sorted_list = sorted(micro_nums)
print()

# s3://microstructure-library/microstructure001/microstructure001.tif

# "0": {
#     "name": "Micrograph 1",
#     "keyword": ["alloy", "nickel"],
#     "category": ["ceramic", "fracture"],
#     "brief_description": "dnfiluahwrfilsuhf",
#     "long_description": "kfjghlsieuhrgliusehrliguhe",
#     "element": ["Cu", "O"],
#     "technique": "SEM",
#     "data_type": "grayscale",
#     "contributor": "Michael",
#     "organisation": "University",
#     "link_doitpoms": "www.hello.com",
#     "data_2D": "snapshot2.png",
#     "data_3D": "snapshot2.png",
#     "preview": "snapshot2.png",
#     "movie": "snapshot2.png"
#   }