class KeywordsNormalizer:
    """
    Utility to normalize string keywords
    - Normalize to lowercase
    - Remove special characters: ?!.,;:_
    - Transform short hands into complete words
    """
    def __init__(self):
        self.keywords_glossary = {
            # ==================== COMPANY & BRANDING ====================
            "cvms": "colour variant",
            "colour variant": "colour variant",
            "color variant": "colour variant",
            "colourvariant": "colour variant",
            "cv multimedia": "colour variant",
            "cvms multimedia": "colour variant",

            # ==================== PRICING ====================
            "hm": "how much",
            "magkano": "how much",
            "magkno": "how much",
            "magkano po": "how much",
            "howmuch": "how much",
            "how much": "how much",
            "price": "price",
            "pricelist": "price list",
            "rate": "price",
            "rates": "price",
            "dp": "downpayment",
            "down": "downpayment",
            "downpayment": "downpayment",

            # ==================== LOCATION ====================
            "loc": "location",
            "location": "location",
            "wer": "where",
            "where": "where",
            "san": "saan",
            "saan": "saan",
            "asan": "saan",
            "tga": "taga",
            "taga": "from",
            "caloocan": "caloocan",
            "calocan": "caloocan",
            "ncr": "ncr",

            # ==================== AVAILABILITY & DATES ====================
            "avail": "available",
            "availability": "available",
            "may slot": "available",
            "slot": "available",
            "open": "available",
            "pwede": "available",
            "pwd": "pwede",
            "may available": "available",
            "meron slot": "available",
            "kelan": "when",
            "kailan": "when",

            # ==================== BOOKING & QUOTE ====================
            "book": "book",
            "booking": "book",
            "reserve": "book",
            "reservation": "book",
            "pa reserve": "book",
            "magbook": "book",
            "magpa book": "book",
            "magpa-book": "book",
            "pa book": "book",
            "how to book": "booking process",
            "paano magbook": "booking process",
            "pano magbook": "booking process",
            "quote": "quote",
            "quotation": "quote",
            "request quote": "quote",

            # ==================== PACKAGES & INCLUSIONS ====================
            "pkg": "package",
            "package": "package",
            "packages": "package",
            "retainer": "retainer",
            "retainer package": "retainer",
            "3 months": "retainer",
            "6 months": "retainer",
            "1 year": "retainer",
            "inclusion": "inclusions",
            "inclusions": "inclusions",
            "ano kasama": "inclusions",
            "kasama": "inclusions",

            # ==================== SERVICES & PORTFOLIO ====================
            "wedding": "wedding",
            "prenup": "prenup",
            "pre nup": "prenup",
            "prenuptial": "prenup",
            "debut": "debut",
            "18th": "debut",
            "js prom": "school event",
            "graduation": "school event",
            "corporate": "corporate",
            "school event": "school event",
            "birthday": "birthday",
            "portrait": "portrait",
            "travel": "travel photography",
            "product shoot": "product photoshoot",
            "product photography": "product photoshoot",
            "event production": "event production",
            "led wall": "led wall",
            "acoustic room": "acoustic room",
            "studio setup": "venue buildout",

            # ==================== PORTFOLIO / GALLERY ====================
            "portfolio": "portfolio",
            "galerry": "portfolio",
            "galery": "portfolio",
            "gallery": "portfolio",
            "portofolio": "portfolio",
            "portpolio": "portfolio",
            "see our work": "portfolio",
            "previous works": "portfolio",
            "samples": "portfolio",
            "examples": "portfolio",

            # ==================== SOCIAL MEDIA & CONTACT ====================
            "ig": "instagram",
            "insta": "instagram",
            "instagram": "instagram",
            "fb": "facebook",
            "facebook": "facebook",
            "messenger": "messenger",
            "yt": "youtube",
            "youtube": "youtube",
            "pm": "private message",
            "dm": "direct message",
            "pa pm": "private message",
            "pa dm": "direct message",

            # ==================== POLITENESS / FILLERS (remove or ignore) ====================
            "po": "",
            "opo": "",
            "sir": "",
            "mam": "",
            "maam": "",
            "ate": "",
            "kuya": "",
            "sis": "",
            "bro": "",
            "brad": "",
            "paps": "",
            "lods": "",
            "boss": "",
            "madam": "",
        }
    
    
    def normalize_string(self, message: str) -> str:
        """
        Removes punctuation, converts to lowercase
        """
        return message.lower().strip().rstrip('?!.,;:_/')  # Prefix for organization
    
    
    def normalize_cache_key(self, message: str) -> str:
        """
        Normalize message for consistent caching
        Removes punctuation, converts to lowercase
        """
        # Remove trailing punctuation and convert to lowercase
        return f"faq:{self.normalize_string(message)}"
    
    
    def normalize_message(self, message: str) -> str:
        """
        Normalize keywords present in the string message
        hm -> how much, magkano -> how much...
        """
        clean_str = self.normalize_string(message)
        normalized_mssg = ""
        for word in clean_str.split():
            normalized_mssg += self.keywords_glossary.get(word, word) + " "
            
        return normalized_mssg.strip()
    
    
kw_norm = KeywordsNormalizer()
