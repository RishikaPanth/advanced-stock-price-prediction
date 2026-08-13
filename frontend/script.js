const predictButton =
    document.getElementById("predictBtn");

const loading =
    document.getElementById("loading");

const errorBox =
    document.getElementById("error");


predictButton.addEventListener(
    "click",
    getPrediction
);


async function getPrediction() {

    loading.classList.remove("hidden");

    errorBox.classList.add("hidden");

    predictButton.disabled = true;

    try {

        const response =
            await fetch("/predict");


        if (!response.ok) {

            throw new Error(
                "Prediction request failed."
            );

        }


        const data =
            await response.json();


        // Update ticker

        document.getElementById(
            "ticker"
        ).textContent = data.ticker;


        // Update prices

        document.getElementById(
            "latestPrice"
        ).textContent =
            `$${data.latest_price.toFixed(2)}`;


        document.getElementById(
            "predictedPrice"
        ).textContent =
            `$${data.predicted_price.toFixed(2)}`;


        // Expected change

        const change =
            data.expected_change_percent;


        const changeElement =
            document.getElementById(
                "expectedChange"
            );


        changeElement.textContent =
            `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`;
        

        // Prediction status

const statusElement =
    document.getElementById(
        "prediction-status"
    );


if (change > 0) {

    statusElement.textContent =
        "↑ Bullish";

    statusElement.className =
        "prediction-status bullish";

}
else if (change < 0) {

    statusElement.textContent =
        "↓ Bearish";

    statusElement.className =
        "prediction-status bearish";

}
else {

    statusElement.textContent =
        "→ Neutral";

    statusElement.className =
        "prediction-status";

}


        // Prediction overview

        document.getElementById(
            "currentBarPrice"
        ).textContent =
            `$${data.latest_price.toFixed(2)}`;


        document.getElementById(
            "predictionBarPrice"
        ).textContent =
            `$${data.predicted_price.toFixed(2)}`;

            // Prediction date

document.getElementById(
    "prediction-date"
).textContent =
    getNextTradingDay();

    }

    catch (error) {

        console.error(error);

        errorBox.textContent =
            "Unable to generate prediction. Please try again.";

        errorBox.classList.remove(
            "hidden"
        );

    }

    finally {

        loading.classList.add(
            "hidden"
        );

        predictButton.disabled = false;

    }
}

function getNextTradingDay() {

    const date = new Date();

    do {

        date.setDate(
            date.getDate() + 1
        );

    }
    while (
        date.getDay() === 0 ||
        date.getDay() === 6
    );

    return date.toLocaleDateString(
        "en-US",
        {
            weekday: "long",
            year: "numeric",
            month: "short",
            day: "numeric"
        }
    );
}


async function loadHistoricalData() {

    try {

        // Get historical prices
        const historyResponse =
            await fetch("/history");

        if (!historyResponse.ok) {
            throw new Error("Failed to load historical data.");
        }

        const history =
            await historyResponse.json();


        // Get latest prediction
        const predictionResponse =
            await fetch("/predict");

        if (!predictionResponse.ok) {
            throw new Error("Failed to load prediction.");
        }

        const prediction =
            await predictionResponse.json();


        // Historical dates
        const labels =
            history.data.map(
                item => item.date
            );


        // Historical prices
        const prices =
            history.data.map(
                item => item.close
            );


        // Add prediction as the next point
        labels.push("Next Trading Day");

        prices.push(null);


        // Separate prediction dataset
        const predictionData =
            new Array(prices.length - 2)
                .fill(null);


        // Connect current price to prediction
        predictionData.push(
            prediction.latest_price
        );

        predictionData.push(
            prediction.predicted_price
        );


        const ctx =
            document
                .getElementById("priceChart")
                .getContext("2d");


        new Chart(ctx, {

            type: "line",

            data: {

                labels: labels,

                datasets: [

                    {
                        label: `${history.ticker} Close Price`,

                        data: prices,

                        borderWidth: 2,

                        pointRadius: 2,

                        tension: 0.2,

                        fill: false
                    },

                    {
                        label: "Next-Day Prediction",

                        data: predictionData,

                        borderWidth: 2,

                        borderDash: [6, 6],

                        pointRadius: 5,

                        tension: 0.2,

                        fill: false
                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                interaction: {

                    mode: "index",

                    intersect: false

                },

                plugins: {

                    legend: {

                        display: true

                    }

                },

                scales: {

                    x: {

                        ticks: {

                            maxTicksLimit: 10

                        }

                    },

                    y: {

                        title: {

                            display: true,

                            text: "Price ($)"

                        }

                    }

                }

            }

        });

    }

    catch (error) {

        console.error(
            "Chart error:",
            error
        );

    }

}


async function loadMetrics() {

    try {

        const response =
            await fetch("/metrics");

        if (!response.ok) {
            throw new Error("Failed to load metrics.");
        }

        const data =
            await response.json();

        document.getElementById("mae")
            .textContent = data.mae.toFixed(2);

        document.getElementById("rmse")
            .textContent = data.rmse.toFixed(2);

        document.getElementById("mape")
            .textContent =
            `${(data.mape * 100).toFixed(2)}%`;

        document.getElementById("r2")
            .textContent = data.r2.toFixed(4);

    }
    catch (error) {

        console.error(
            "Metrics error:",
            error
        );

    }
}


// Generate prediction when page loads

getPrediction();
loadHistoricalData();
loadMetrics();