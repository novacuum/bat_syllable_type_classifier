import requests

#https://eudat.eu/services/userdoc/b2share-http-rest-api#upload-file-into-draft-record

B2SHARE_HOST = 'trng-b2share.eudat.eu'
FILE_BUCKET_ID = 'c5907f7e-7594-459e-994b-34bbbaebf55d'
ACCESS_TOKEN = ''


def get_file_list():
    res = requests.get(
        url=f'https://{B2SHARE_HOST}/api/files/{FILE_BUCKET_ID}?access_token={ACCESS_TOKEN}',
        headers = {
            'Content-Type': 'application/octet-stream'
            , 'User-Agent': 'Python'
            , 'Accept': 'application/json'
        }
    )


def upload_file(file, filename):
    res = requests.put(
        url=f'https://{B2SHARE_HOST}/api/files/{FILE_BUCKET_ID}/{filename}?access_token={ACCESS_TOKEN}',
        data=open(file, 'rb').read(),
        headers={
            'Content-Type': 'application/octet-stream'
            , 'User-Agent': 'Python'
            , 'Accept': 'application/json'
        })
