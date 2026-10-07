const modelTitle = {

    text:
        "Text-only",

    audio:
        "Audio-only",

    fusion:
        "Attention Fusion",
};


const statusBadge =
    document.getElementById(
        "statusBadge"
    );


const metricsCards =
    document.getElementById(
        "metricsCards"
    );


const datasetInfo =
    document.getElementById(
        "datasetInfo"
    );


const trainBtn =
    document.getElementById(
        "trainBtn"
    );


const trainMessage =
    document.getElementById(
        "trainMessage"
    );


const chartInstances = {};


/*
==================================
METRIC CARDS
==================================
*/

function metricCard(
    name,
    metrics
) {

    return `
        <article class="card">

            <h3>
                ${modelTitle[name]}
            </h3>

            <div class="metric">

                <span>
                    Accuracy
                </span>

                <strong>
                    ${(
            metrics.accuracy
            * 100
        ).toFixed(2)
        }%
                </strong>

            </div>

            <div class="metric">

                <span>
                    Macro-F1
                </span>

                <strong>
                    ${(
            metrics.macro_f1
            * 100
        ).toFixed(2)
        }%
                </strong>

            </div>

            <div class="metric">

                <span>
                    Validation loss
                </span>

                <strong>
                    ${metrics
            .validation_loss
            .toFixed(4)
        }
                </strong>

            </div>

            <div class="metric">

                <span>
                    Train time
                </span>

                <strong>
                    ${metrics
            .train_seconds
            .toFixed(1)
        } s
                </strong>

            </div>

        </article>
    `;
}


/*
==================================
CHART HELPERS
==================================
*/

function destroyChart(
    id
) {

    if (
        chartInstances[id]
    ) {

        chartInstances[
            id
        ].destroy();

        delete chartInstances[
            id
        ];
    }
}


function renderLineChart(
    canvasId,
    history,
    trainKey,
    validationKey,
    yTitle
) {

    const canvas =
        document.getElementById(
            canvasId
        );


    if (
        !canvas
        ||
        typeof Chart === "undefined"
        ||
        !history
    ) {
        return;
    }


    destroyChart(
        canvasId
    );


    const firstHistory =
        Object.values(
            history
        )[0]
        || [];


    const labels =
        firstHistory.map(
            point =>
                `Epoch ${point.epoch}`
        );


    const palette = {

        text: [
            "#6d8cff",
            "#9bb0ff",
        ],

        audio: [
            "#f19a66",
            "#ffc09b",
        ],

        fusion: [
            "#64d8a7",
            "#9af0cc",
        ],
    };


    const datasets = [];


    for (
        const [
            modelName,
            points
        ]
        of Object.entries(
            history
        )
    ) {

        /*
        Для loss показываем
        train и validation.
        */

        if (
            trainKey
        ) {

            datasets.push({

                label:
                    `${modelTitle[modelName]} train`,

                data:
                    points.map(
                        point =>
                            point[
                            trainKey
                            ]
                    ),

                borderColor:
                    palette[
                    modelName
                    ][1],

                backgroundColor:
                    palette[
                    modelName
                    ][1],

                borderDash: [
                    6,
                    5,
                ],

                tension:
                    0.25,
            });
        }


        datasets.push({

            label:
                `${modelTitle[modelName]} validation`,

            data:
                points.map(
                    point =>
                        point[
                        validationKey
                        ]
                ),

            borderColor:
                palette[
                modelName
                ][0],

            backgroundColor:
                palette[
                modelName
                ][0],

            borderWidth:
                modelName === "fusion"
                    ? 3
                    : 2,

            tension:
                0.25,
        });
    }


    chartInstances[
        canvasId
    ] = new Chart(
        canvas,
        {

            type:
                "line",

            data: {
                labels,
                datasets,
            },

            options: {

                responsive:
                    true,

                maintainAspectRatio:
                    false,

                interaction: {

                    mode:
                        "index",

                    intersect:
                        false,
                },

                plugins: {

                    legend: {

                        labels: {

                            color:
                                "#dbe4f4",
                        },
                    },
                },

                scales: {

                    x: {

                        ticks: {

                            color:
                                "#aeb9cf",
                        },

                        grid: {

                            color:
                                "rgba(174,185,207,0.12)",
                        },
                    },

                    y: {

                        title: {

                            display:
                                true,

                            text:
                                yTitle,

                            color:
                                "#aeb9cf",
                        },

                        ticks: {

                            color:
                                "#aeb9cf",
                        },

                        grid: {

                            color:
                                "rgba(174,185,207,0.12)",
                        },
                    },
                },
            },
        }
    );
}


/*
==================================
TRAINING CHARTS
==================================
*/

function renderTrainingCharts(
    history
) {

    if (
        !history
    ) {
        return;
    }


    renderLineChart(

        "lossChart",

        history,

        "train_loss",

        "validation_loss",

        "Loss"
    );


    renderLineChart(

        "accuracyChart",

        history,

        null,

        "validation_accuracy",

        "Accuracy"
    );


    renderLineChart(

        "f1Chart",

        history,

        null,

        "validation_macro_f1",

        "Macro-F1"
    );
}


/*
==================================
RESULTS
==================================
*/

function renderResults(
    data
) {

    if (
        !data.trained
        &&
        !data.metrics
    ) {

        statusBadge.textContent =
            "не обучено";

        statusBadge.classList.remove(
            "ok"
        );

        metricsCards.innerHTML = `
            <div class="muted">
                Сначала запустите
                обучение моделей.
            </div>
        `;

        datasetInfo.textContent =
            "";

        return;
    }


    statusBadge.textContent =
        "обучено";


    statusBadge.classList.add(
        "ok"
    );


    metricsCards.innerHTML =
        Object.entries(
            data.metrics
        )

            .map(
                (
                    [
                        name,
                        metrics
                    ]
                ) =>
                    metricCard(
                        name,
                        metrics
                    )
            )

            .join("");


    datasetInfo.textContent =

        `Train: ${data.train_size} | `

        + `Validation: ${data.validation_size} | `

        + `Epochs: ${data.epochs} | `

        + `Batch size: ${data.batch_size} | `

        + `Device: ${data.device} | `

        + `Пустых train-транскрипций: `
        + `${data.empty_train_transcripts}`;


    renderTrainingCharts(
        data.history
    );
}


/*
==================================
LOAD OLD RESULTS
==================================
*/

async function loadResults() {

    const response =
        await fetch(
            "/api/results"
        );


    const data =
        await response.json();


    renderResults(
        data
    );
}


/*
==================================
TRAINING
==================================
*/

trainBtn.addEventListener(
    "click",
    async () => {

        trainBtn.disabled =
            true;


        trainMessage.textContent =
            "Обучение идёт. "
            + "Это может занять "
            + "несколько минут.";


        const epochs =
            document
                .getElementById(
                    "epochs"
                )
                .value;


        const batchSize =
            document
                .getElementById(
                    "batchSize"
                )
                .value;


        try {

            const url =
                `/api/train`
                + `?epochs=${encodeURIComponent(
                    epochs
                )
                }`
                + `&batch_size=${encodeURIComponent(
                    batchSize
                )
                }`;


            const response =
                await fetch(
                    url,
                    {
                        method:
                            "POST",
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
            ) {

                throw new Error(
                    data.detail
                    ||
                    "Training failed"
                );
            }


            renderResults(
                data
            );


            trainMessage.textContent =
                "Готово. "
                + "Все три модели "
                + "обучены "
                + "и сохранены.";

        }

        catch (
        error
        ) {

            trainMessage.textContent =
                `Ошибка: ${error.message
                }`;
        }

        finally {

            trainBtn.disabled =
                false;
        }
    }
);


/*
==================================
PREDICTION
==================================
*/

document
    .getElementById(
        "predictForm"
    )
    .addEventListener(
        "submit",
        async (
            event
        ) => {

            event.preventDefault();


            const audio =
                document
                    .getElementById(
                        "audioFile"
                    )
                    .files[0];


            const text =
                document
                    .getElementById(
                        "transcript"
                    )
                    .value;


            const form =
                new FormData();


            form.append(
                "audio",
                audio
            );


            form.append(
                "text",
                text
            );


            const response =
                await fetch(
                    "/api/predict",
                    {

                        method:
                            "POST",

                        body:
                            form,
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
            ) {

                document
                    .getElementById(
                        "predictionCards"
                    )
                    .innerHTML = `

                <div class="message">
                    Ошибка:
                    ${data.detail}
                </div>
            `;

                return;
            }


            /*
            Prediction cards
            */

            document
                .getElementById(
                    "predictionCards"
                )
                .innerHTML =

                Object.entries(
                    data
                )

                    .map(
                        (
                            [
                                name,
                                result
                            ]
                        ) => `

                    <article class="card">

                        <h3>
                            ${modelTitle[
                            name
                            ]
                            }
                        </h3>

                        <div class="metric">

                            <span>
                                Prediction
                            </span>

                            <strong>
                                ${result.label
                            }
                            </strong>

                        </div>

                        <div class="metric">

                            <span>
                                Confidence
                            </span>

                            <strong>

                                ${(
                                result.confidence
                                * 100
                            )
                                .toFixed(
                                    2
                                )
                            }%

                            </strong>

                        </div>

                    </article>
                `
                    )

                    .join("");


            /*
            Attention visualization
            */

            const attention =
                data.fusion
                    ?.attention;


            if (
                attention
            ) {

                document
                    .getElementById(
                        "attentionBox"
                    )
                    .innerHTML = `

                <div class="attention">

                    <strong>
                        Attention Fusion:
                        вклад модальностей
                    </strong>


                    <p>
                        Text:
                        ${(
                        attention.text
                        * 100
                    )
                        .toFixed(
                            1
                        )
                    }%
                    </p>


                    <div class="bar">

                        <span
                            style="
                                width:
                                ${attention.text
                    * 100
                    }%
                            "
                        ></span>

                    </div>


                    <p>
                        Audio:
                        ${(
                        attention.audio
                        * 100
                    )
                        .toFixed(
                            1
                        )
                    }%
                    </p>


                    <div class="bar">

                        <span
                            style="
                                width:
                                ${attention.audio
                    * 100
                    }%
                            "
                        ></span>

                    </div>

                </div>
            `;
            }
        }
    );


/*
==================================
INITIAL PAGE LOAD
==================================
*/

loadResults();