import boto3
import csv
with open('rekognition_key.csv','r') as input:
    next(input)
    reader=csv.reader(input)
    for line in reader:
        access_key_id=line[2]
        secret_key_id=line[3]


client = boto3.client('rekognition',
                      aws_access_key_id=access_key_id,
                      aws_secret_access_key=secret_key_id,
                      region_name='us-east-1'
                      )
def compareFaces():
    source='image/muru.jpg'   # three_peoples   muru  joker  girl properimage1 kids  aadha, pan , photo, payslip , company name address...
    target='image/muru.jpg'
    with open(source,'rb') as source_image:
        source_bytes=source_image.read()
    with open(target,'rb') as target_image:
        target_bytes=target_image.read()

    response = client.compare_faces(
    SourceImage={
        'Bytes': source_bytes},
    TargetImage={
        'Bytes': target_bytes
    } )
    #print()
    for key,value in response.items():
        if key in ('FaceMatches','UnmatchedFaces'):
            print(key)
            for att in value:
                print(att)
                print()
    '''for faceMatch in response['FaceMatches']:
        position = faceMatch['Face']['BoundingBox']
        similarity = str(faceMatch['Similarity'])
        print('The face at ' +
               str(position['Left']) + ' ' +
               str(position['Top']) +
               ' matches with ' + similarity + '% confidence')'''
def detectText():
    photo='image/aadhar.jpg'   # pan aadhar
    with open(photo,'rb') as source_image:
        source_bytes=source_image.read()
    response=client.detect_text(Image={'Bytes': source_bytes})
    print(response)
    #print()
    #return len(response['FaceDetails'])

def detectFaces():
    photo='image/muru.jpg'   # three_peoples   muru  joker  girl properimage1 kids  6100001
    with open(photo,'rb') as source_image:
        source_bytes=source_image.read()
    #print(source_bytes)
    outF = open("my_bytes.txt", "wb")
    outF.write(source_bytes)
    outF.close()
    #print(source_bytes)
    print()
    response=client.detect_faces(Image={'Bytes': source_bytes},Attributes=['ALL'])
    print(response['FaceDetails'][0]['Gender'])
    #print()
    confidence=response['FaceDetails'][0]['Gender']['Confidence']
    if confidence>80:
        print("Valid Faces...")
    
    print("Number of faces in a given images is : ",len(response['FaceDetails']))
    return len(response['FaceDetails'])

detectFace_result=detectFaces()
#compareFace_result=compareFaces()
#detect_text=detectText()

'''photo='image/small.jpg'   # three_peoples   muru  joker  girl properimage1 kids
with open(photo,'rb') as source_image:
    source_bytes=source_image.read()

response=client.detect_faces(Image={'Bytes': source_bytes},Attributes=['ALL'])
#print(response['FaceDetails'])
#print()
print("Number of faces in a given images is : ",len(response['FaceDetails']))'''
