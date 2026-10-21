"""
Email client integration.

Provides a provider-agnostic email client for the
Enterprise Legal Document Intelligence AI platform.

Supported delivery modes:
- SMTP
- Development/mock mode

Business logic should remain in notification_service.py.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Sequence

from backend.config import settings

logger = logging.getLogger(__name__)


class EmailClientError(Exception):
    """Base exception for email delivery failures."""


class EmailConfigurationError(EmailClientError):
    """Raised when email configuration is incomplete."""


class EmailDeliveryError(EmailClientError):
    """Raised when an email cannot be delivered."""


@dataclass(frozen=True)
class EmailPayload:
    """Normalized email payload."""

    to: str | Sequence[str]
    subject: str
    body: str
    html_body: str | None = None
    cc: Sequence[str] | None = None
    bcc: Sequence[str] | None = None
    reply_to: str | None = None


class EmailClient:
    """
    Provider-agnostic email delivery client.

    The notification service should depend on this class rather
    than directly interacting with SMTP.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
        use_tls: bool | None = None,
        use_ssl: bool | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.host = host or getattr(settings, "SMTP_HOST", None)
        self.port = port or getattr(settings, "SMTP_PORT", 587)

        self.username = username or getattr(
            settings,
            "SMTP_USERNAME",
            None,
        )

        self.password = password or getattr(
            settings,
            "SMTP_PASSWORD",
            None,
        )

        self.from_email = from_email or getattr(
            settings,
            "EMAIL_FROM",
            None,
        )

        self.from_name = from_name or getattr(
            settings,
            "EMAIL_FROM_NAME",
            "Enterprise Legal Intelligence",
        )

        self.use_tls = (
            use_tls
            if use_tls is not None
            else getattr(settings, "SMTP_USE_TLS", True)
        )

        self.use_ssl = (
            use_ssl
            if use_ssl is not None
            else getattr(settings, "SMTP_USE_SSL", False)
        )

        self.enabled = (
            enabled
            if enabled is not None
            else getattr(settings, "EMAIL_ENABLED", True)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        *,
        to: str | Sequence[str],
        subject: str,
        body: str,
        html_body: str | None = None,
        cc: Sequence[str] | None = None,
        bcc: Sequence[str] | None = None,
        reply_to: str | None = None,
    ) -> bool:
        """
        Send an email.

        Returns True after successful delivery.

        Raises:
            EmailConfigurationError:
                Invalid or incomplete configuration.

            EmailDeliveryError:
                Provider rejected or failed to deliver the message.
        """
        payload = EmailPayload(
            to=to,
            subject=subject,
            body=body,
            html_body=html_body,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
        )

        self._validate_configuration(payload)

        if not self.enabled:
            logger.info(
                "Email delivery disabled",
                extra={
                    "subject": subject,
                    "recipients": self._normalize_addresses(to),
                },
            )
            return True

        message = self._build_message(payload)

        try:
            self._send_smtp(message, payload)

            logger.info(
                "Email delivered",
                extra={
                    "subject": subject,
                    "recipients": self._normalize_addresses(to),
                },
            )

            return True

        except EmailClientError:
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected email delivery failure",
                extra={"subject": subject},
            )

            raise EmailDeliveryError(
                "Unexpected email delivery failure"
            ) from exc

    def send_text(
        self,
        *,
        to: str | Sequence[str],
        subject: str,
        body: str,
        **kwargs,
    ) -> bool:
        """Send a plain-text email."""
        return self.send(
            to=to,
            subject=subject,
            body=body,
            **kwargs,
        )

    def send_html(
        self,
        *,
        to: str | Sequence[str],
        subject: str,
        html_body: str,
        text_body: str = "",
        **kwargs,
    ) -> bool:
        """Send a multipart text + HTML email."""
        return self.send(
            to=to,
            subject=subject,
            body=text_body,
            html_body=html_body,
            **kwargs,
        )

    def health_check(self) -> bool:
        """
        Verify that the configured SMTP server can be reached.

        Does not send an email.
        """
        self._validate_base_configuration()

        try:
            smtp = self._create_smtp_connection()

            try:
                smtp.noop()

            finally:
                smtp.quit()

            return True

        except Exception as exc:
            logger.exception("SMTP health check failed")

            raise EmailDeliveryError(
                "SMTP health check failed"
            ) from exc

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_message(
        self,
        payload: EmailPayload,
    ) -> EmailMessage:
        """Build a standards-compliant email message."""
        message = EmailMessage()

        message["From"] = self._format_sender()

        recipients = self._normalize_addresses(payload.to)

        message["To"] = ", ".join(recipients)
        message["Subject"] = payload.subject

        if payload.cc:
            message["Cc"] = ", ".join(
                self._normalize_addresses(payload.cc)
            )

        if payload.reply_to:
            message["Reply-To"] = payload.reply_to

        message.set_content(payload.body)

        if payload.html_body:
            message.add_alternative(
                payload.html_body,
                subtype="html",
            )

        return message

    def _format_sender(self) -> str:
        """Return the configured From header."""
        if not self.from_email:
            raise EmailConfigurationError(
                "EMAIL_FROM is not configured."
            )

        if self.from_name:
            return f"{self.from_name} <{self.from_email}>"

        return self.from_email

    # ------------------------------------------------------------------
    # SMTP
    # ------------------------------------------------------------------

    def _send_smtp(
        self,
        message: EmailMessage,
        payload: EmailPayload,
    ) -> None:
        """Send an email through SMTP."""
        smtp = self._create_smtp_connection()

        try:
            if self.username and self.password:
                smtp.login(
                    self.username,
                    self.password,
                )

            recipients = (
                self._normalize_addresses(payload.to)
                + self._normalize_addresses(payload.cc)
                + self._normalize_addresses(payload.bcc)
            )

            smtp.send_message(
                message,
                from_addr=self.from_email,
                to_addrs=recipients,
            )

        except smtplib.SMTPException as exc:
            logger.exception("SMTP delivery failed")

            raise EmailDeliveryError(
                "SMTP delivery failed."
            ) from exc

        finally:
            try:
                smtp.quit()
            except Exception:
                logger.debug(
                    "SMTP connection close failed",
                    exc_info=True,
                )

    def _create_smtp_connection(self):
        """Create the appropriate SMTP connection."""
        self._validate_base_configuration()

        if self.use_ssl:
            smtp = smtplib.SMTP_SSL(
                self.host,
                self.port,
                timeout=30,
            )
        else:
            smtp = smtplib.SMTP(
                self.host,
                self.port,
                timeout=30,
            )

            if self.use_tls:
                smtp.starttls()

        return smtp

    # ------------------------------------------------------------------
    # Validation / helpers
    # ------------------------------------------------------------------

    def _validate_configuration(
        self,
        payload: EmailPayload,
    ) -> None:
        """Validate email configuration and payload."""
        self._validate_base_configuration()

        if not payload.subject.strip():
            raise EmailConfigurationError(
                "Email subject cannot be empty."
            )

        if not payload.body.strip() and not payload.html_body:
            raise EmailConfigurationError(
                "Email must contain a text or HTML body."
            )

        if not self._normalize_addresses(payload.to):
            raise EmailConfigurationError(
                "At least one recipient is required."
            )

    def _validate_base_configuration(self) -> None:
        """Validate SMTP configuration."""
        if not self.host:
            raise EmailConfigurationError(
                "SMTP_HOST is not configured."
            )

        if not self.port:
            raise EmailConfigurationError(
                "SMTP_PORT is not configured."
            )

        if not self.from_email:
            raise EmailConfigurationError(
                "EMAIL_FROM is not configured."
            )

    @staticmethod
    def _normalize_addresses(
        addresses: str | Sequence[str] | None,
    ) -> list[str]:
        """Normalize recipient collections."""
        if not addresses:
            return []

        if isinstance(addresses, str):
            return [
                address.strip()
                for address in addresses.split(",")
                if address.strip()
            ]

        return [
            address.strip()
            for address in addresses
            if address and address.strip()
        ]


def get_email_client() -> EmailClient:
    """
    Application-level email client factory.

    Keeping construction here makes it easy to replace SMTP
    with another provider later without changing notification logic.
    """
    return EmailClient()