import requests  # to get image from the web
from bs4 import BeautifulSoup

text_to_key = {
    "name": "Micrograph 1",
    "keyword": "Keywords",
    "category": "Categories",
    "brief_description": "Brief description",
    "long_description": "Further information",
    "element": "Composition",
    "technique": "Technique",
    "data_type": "grayscale",
    "contributor": "Contributor",
    "organisation": "Organisation"
  }

def find_and_print_text(soup_block, header, return_list=False):
    """
    :param soup_block: The beautifulsoup object
    :param header: The header for the text to be returned
    :return: (Boolean, text) where the Boolean is true iff the header exists
    in the soup block, and the appropriate text if it exist.
    """
    if soup_block.find(text=header):
        if soup_block.findNext().find(text=header):
            print(f"{header}: {block.findNext('dd').text}")


for i in range(1, 21):
    # Get the url:
    record_url = f'https://www.doitpoms.ac.uk/miclib/full_record.php?id={i}'
    record_request = requests.get(record_url)
    soup = BeautifulSoup(record_request.content, 'html.parser')
    soup_info = soup.find('div', class_='col-md-8')
    soup_info = soup_info.findAll()
    print(f'Micrograph number: {i}')
    for block in soup_info:
        keys = ['Keywords', 'Brief description', 'Further information', ]
        find_and_print_text(block, 'Keywords')
        find_and_print_text(block, 'Brief description')
        find_and_print_text(block, 'Further information')
    print()


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