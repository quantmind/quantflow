from pydantic import BaseModel, Field

from .numbers import ZERO, Decimal, DecimalNumber


class Price(BaseModel):
    """Represents the bid/ask price of a security,
    which can be a spot price, forward price or option price
    """

    bid: DecimalNumber = Field(description="Bid price")
    ask: DecimalNumber = Field(description="Ask price")

    @property
    def mid(self) -> Decimal:
        """Calculate the mid price by averaging the bid and ask prices"""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate the bid-ask spread"""
        return self.ask - self.bid

    @property
    def bp_spread(self) -> Decimal:
        """Bid-ask spread in basis points, calculated as spread divided by mid
        price and multiplied by 10000"""
        mid = self.mid
        if mid > ZERO:
            return round(10000 * self.spread / mid, 2)
        else:
            return Decimal("inf")

    def is_valid(self) -> bool:
        """Check if the price is valid, which means the bid is less than
        or equal to the ask"""
        return self.bid <= self.ask


class PriceVolume(Price):
    """Base class for price with volume and open interest"""

    open_interest: DecimalNumber = Field(
        default=ZERO, description="Open interest of the security"
    )
    volume: DecimalNumber = Field(default=ZERO, description="Volume of the security")
