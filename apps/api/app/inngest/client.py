import logging
import inngest

from app.core.config import settings

logger = logging.getLogger(__name__)

inngest_client = inngest.Inngest(
    app_id=settings.inngest_app_id,
    api_base_url=settings.inngest_api_base_url or None,
    event_api_base_url=settings.inngest_event_api_base_url or None,
    event_key=settings.inngest_event_key or None,
    signing_key=settings.inngest_signing_key or None,
    is_production=not settings.inngest_is_dev,
)
