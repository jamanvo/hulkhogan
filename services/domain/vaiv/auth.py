import requests
from django.conf import settings


def get_access_token() -> str | None:
    access_token = None
    try:
        token_response = requests.post(
            settings.KEYCLOAK_TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id": settings.CLIENT_ID,
                "client_secret": settings.CLIENT_SECRET,
                "username": settings.VAIV_USERNAME,
                "password": settings.VAIV_PASSWORD,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )

        if token_response.status_code != 200:
            return None

        token_json = token_response.json()
        access_token = token_json.get("access_token")
        if not access_token:
            return None
    except Exception as e:
        print("VAIV Auth Error: ", e)

    return access_token
