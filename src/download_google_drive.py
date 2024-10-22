
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
import io

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive_api():
    """
    Authenticate and build the Google Drive API service.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def download_files_from_folder(service, folder_id, download_dir):
    """
    Downloads all files from a specific Google Drive folder.

    Args:
    - service: Authenticated Google Drive API service instance.
    - folder_id: The ID of the Google Drive folder.
    - download_dir: Local directory where the files will be downloaded.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # List all files in the folder
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)"
    ).execute()

    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        for item in items:
            file_id = item['id']
            file_name = item['name']
            print(f"Downloading {file_name}...")

            request = service.files().get_media(fileId=file_id)
            file_path = os.path.join(download_dir, file_name)

            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    print(f"Download {file_name}: {int(status.progress() * 100)}%")

