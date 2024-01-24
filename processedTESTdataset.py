import os
label_dict = {'药物':'DRUG',
              '解剖部位':'BODY',
              '疾病和诊断':'DISEASES',
              '影像检查':'EXAMINATIONS',
              '实验室检验':'TEST',
              '手术':'TREATMENT'}


def sentence2BIOlabel(sentence, label_from_file):
    """ BIO Tagging """
    sentence_label = ['O']*len(sentence)
    if label_from_file=='':
        return sentence_label
    
    for line in label_from_file.split('\n'):
        
        entity_info = line.strip().split('\t')
        start_index = int(entity_info[1])     
        end_index = int(entity_info[2])      
        entity_label = label_dict[entity_info[3]]      
        # Frist entity: B-xx
        sentence_label[start_index] = 'B-'+entity_label
        # Other: I-xx
        for i in range(start_index+1, end_index):
            sentence_label[i] = 'I-'+entity_label
    return sentence_label

def loadRawData(fileDirect, Name):
    """ Loading raw data and tagging """
    sentence_list = []

    fileName = fileDirect + Name
    with open(fileName, encoding='utf-8') as f:
        content = f.read().strip()    
    
    sentence_list.append(content)
    
    return sentence_list
    

def Save_data(filename, texts):
  """ Processing to files in neeed format """
  with open(filename, 'w', encoding='utf-8') as f:
    for sent in texts:
        size = len(sent)
        for i in range(size):
          f.write(sent[i])
          f.write('\n')

# Test data
NameOri = 'data-test-2-original.txt'
sentence_list_test = loadRawData('D:/Liuuu/Undergraduate/GraduationProject/Codes/BertBilstmCRF/ChineseMedicalEntityRecognitionmaster/CCKS_2019_Task1/data_test/',NameOri)

# Split dataset
t_words = [list(sent) for sent in sentence_list_test]
test_texts = t_words

Path = 'D:/Liuuu/Undergraduate/GraduationProject/Codes/BertBilstmCRF/ChineseMedicalEntityRecognitionmaster/CCKS_2019_Task1/processed_data/'
Name = 'test2.txt'

TEST = Path + Name
# Obtain testing files
Save_data(TEST, test_texts)