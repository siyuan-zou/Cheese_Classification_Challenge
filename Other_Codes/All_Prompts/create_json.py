import json

def create_json():
    with open("/users/eleves-b/2022/siyuan.zou/DL_SiyuanZou/Chellenge_Cheese/prompts/production.txt", "r") as f:
        content = f.read()
    
    # Split the text into sections based on double newline, assuming each cheese description is separated this way
    sections = content.strip().split('\n\n')
    
    # Process each section to extract the name and description
    data = {}
    for section in sections:
        lines = section.split('\n')
        cheese_name = lines[0].strip().upper() 
        description = ' '.join(line.strip() for line in lines[1:])

        # Create a dictionary with uppercase keys
        data.update({cheese_name:description})
    
    # print(data['CHÈVRE'])
    # Convert the list of dictionaries to JSON
    json_data = json.dumps(data, indent=4)
    with open("/users/eleves-b/2022/siyuan.zou/DL_SiyuanZou/Chellenge_Cheese/prompts/production_l.json", 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

if __name__ == "__main__":
    create_json()
    
