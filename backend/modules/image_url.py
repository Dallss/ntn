import re
import requests
from bs4 import BeautifulSoup

class FragranticaImageExtractor:
    CDN_PREFIX = "https://fimgs.net/mdimg/"

    # ---- Browser-like headers to avoid 403
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.fragrantica.com/",
        "Connection": "keep-alive",
    }

    @staticmethod
    def is_fragrantica_url(url: str) -> bool:
        return "fragrantica.com/perfume/" in url

    @staticmethod
    def get_perfume_image(url: str) -> tuple[str | None, str]:
        """
        Returns a single image URL for a Fragrantica perfume page,
        or a string explaining why it failed.
        """

        if not FragranticaImageExtractor.is_fragrantica_url(url):
            return None, "URL is not a fragrantica perfume page"

        try:
            response = requests.get(url, headers=FragranticaImageExtractor.HEADERS, timeout=10)
            response.raise_for_status()
        except requests.HTTPError as e:
            return None, f"http error: {e}"
        except Exception as e:
            return None, f"request failed: {e}"

        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # ---- 1. Try <source type="image/avif">
        sources = soup.find_all("source", type="image/avif")
        if sources:
            for source in sources:
                srcset = source.get("srcset", "")
                if not srcset:
                    continue

                for part in srcset.split(","):
                    img_url = part.strip().split(" ")[0]
                    if img_url.startswith(FragranticaImageExtractor.CDN_PREFIX):
                        return img_url, "ok (from srcset)"

        reason = "no avif <source> tags found or no fimgs.net urls inside them"

        # ---- 2. Regex fallback
        matches = re.findall(
            r"https://fimgs\.net/mdimg/[^\s\"']+?\.(?:avif|webp|jpg|png)",
            html
        )
        if matches:
            return matches[0], "ok (from regex)"

        # ---- 3. Give up
        return None, reason
