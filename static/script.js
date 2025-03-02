fetch('/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        stockTicker: stockTicker,
        strikePrice: strikePrice,
        daysUntilExpiration: daysUntilExpiration
    })
})
.then(response => response.json())
.then(data => {
    // Display the option price
    document.getElementById('result').innerText = "Option Price: " + data.optionPrice;
})
.catch(error => console.error('Error:', error));