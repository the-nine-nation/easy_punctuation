import unicodedata
import os
def is_chinese(char):
    if 'CJK' in unicodedata.name(char):
        return True
    else:
        return False
def is_english(char):
    return char.isalpha()
def is_num(char):
    return char.isdigit()


punc_list=["，","。","：","！","？",".","?","!",":",";","；","、"]
endlist=["。","！","？",".","?","!"]
def clean_text(src_txt,tag_txt):
    clear_text = []
    with open(src_txt, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line=line.replace("\n","")#删除换行符
            line=line.strip()#删除空格
            line = line.replace('\n', '')
            line = line.replace('①', '')
            line = line.replace('②', '')
            line = line.replace('③', '')
            line = line.replace('④', '')
            line = line.replace('⑤', '')
            line = line.replace('⑥', '')
            line = line.replace('⑦', '')
            line = line.replace('⑧', '')
            line = line.replace('⑨', '')
            line = line.replace('⑩', '')
            line = line.replace('。。。', '。')
            line = line.replace('啊……', '啊！')
            line = line.replace('？！', '？')
            line = line.replace('！！！', '！')
            if len(line)>2:
                if (line[-1] in endlist) and (line[1] not in punc_list):#判断末尾是否存在设想标点
                    clear_line=""
                    last_pun=None
                    for i in line:
                        if is_num(i) or is_english(i) or is_num(i) or i==" " or ((i in punc_list) and i!=last_pun):
                            clear_line+=i
                        last_pun=i
                    if clear_line!="":
                        clear_text.append(clear_line)
    with open(tag_txt, 'w', encoding='utf-8') as fps:
        for i in clear_text:
            fps.writelines(i+"\n")
for root, dirs, files in os.walk("./txt_clear"):
    for file in files:
        txt_path=os.path.join(root, file)
        #tag_path=os.path.join("./txt_clear",file)
        clean_text(txt_path,txt_path)