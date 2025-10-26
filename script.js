
class HousePricePredictor {
    constructor() {
        this.parameters = null;
        this.normStats = null;
    }

    // Load model parameters from JSON file
    async loadModel(modelPath = 'model_parameters.json') {
        try {
            const response = await fetch(modelPath);
            this.parameters = await response.json();
            console.log('✅ Model parameters loaded successfully');
            return true;
        } catch (error) {
            console.error('❌ Error loading model parameters:', error);
            return false;
        }
    }

    // Load normalization statistics from JSON file
    async loadNormalization(normPath = 'normalization.json') {
        try {
            const response = await fetch(normPath);
            this.normStats = await response.json();
            console.log('✅ Normalization stats loaded successfully');
            return true;
        } catch (error) {
            console.error('❌ Error loading normalization stats:', error);
            return false;
        }
    }

    // Initialize - load both model and normalization files
    async initialize(modelPath = 'model_parameters.json', normPath = 'normalization.json') {
        const modelLoaded = await this.loadModel(modelPath);
        const normLoaded = await this.loadNormalization(normPath);
        
        if (modelLoaded && normLoaded) {
            console.log('✅ Predictor initialized successfully');
            return true;
        }
        return false;
    }

    // Leaky ReLU activation function
    leakyReLU(z, alpha = 0.01) {
        if (Array.isArray(z)) {
            return z.map(val => val > 0 ? val : alpha * val);
        }
        return z > 0 ? z : alpha * z;
    }

    // Matrix multiplication
    matrixMultiply(A, B) {
        const rowsA = A.length;
        const colsA = A[0].length;
        const colsB = B[0].length;
        
        const result = [];
        for (let i = 0; i < rowsA; i++) {
            result[i] = [];
            for (let j = 0; j < colsB; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    // Add bias to matrix
    addBias(matrix, bias) {
        return matrix.map(row => 
            row.map((val, idx) => val + bias[0][idx])
        );
    }

    // Forward propagation through the neural network
    forwardPropagation(X) {
        const w1 = this.parameters.w1;
        const w2 = this.parameters.w2;
        const w3 = this.parameters.w3;
        const w4 = this.parameters.w4;
        
        const b1 = this.parameters.b1;
        const b2 = this.parameters.b2;
        const b3 = this.parameters.b3;
        const b4 = this.parameters.b4;

        // Layer 1
        let z1 = this.matrixMultiply(X, w1);
        z1 = this.addBias(z1, b1);
        let A1 = z1.map(row => this.leakyReLU(row));

        // Layer 2
        let z2 = this.matrixMultiply(A1, w2);
        z2 = this.addBias(z2, b2);
        let A2 = z2.map(row => this.leakyReLU(row));

        // Layer 3
        let z3 = this.matrixMultiply(A2, w3);
        z3 = this.addBias(z3, b3);
        let A3 = z3.map(row => this.leakyReLU(row));

        // Layer 4 (output layer - linear activation)
        let z4 = this.matrixMultiply(A3, w4);
        z4 = this.addBias(z4, b4);
        let A4 = z4; // Linear activation

        return A4;
    }

    // Preprocess input data from your form
    preprocessInput(formData) {
        // Convert furnishing status to one-hot encoding
        const furnished = formData.furnishing === 'furnished' ? 1 : 0;
        const semiFurnished = formData.furnishing === 'semi-furnished' ? 1 : 0;
        const unfurnished = formData.furnishing === 'unfurnished' ? 1 : 0;

        // Create feature array in exact order as training data (13 features total)
        // Order: area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, 
        //        hotwaterheating, airconditioning, parking, prefarea, furnished, semi-furnished, unfurnished
        const features = [
            formData.area,
            formData.bedrooms,
            formData.bathrooms,
            formData.stories,
            formData.mainroad,
            formData.guestroom,
            formData.basement,
            formData.hotwater,  // hotwaterheating
            formData.ac,        // airconditioning
            formData.parking,
            formData.prefarea,
            furnished,
            semiFurnished,
            unfurnished
        ];

        return features;
    }

    // Normalize input features
    normalizeInput(features) {
        const meanX = this.normStats.mean_x;
        const stdX = this.normStats.std_x;

        const normalized = features.map((val, idx) => {
            return (val - meanX[idx]) / stdX[idx];
        });

        return [normalized]; // Return as 2D array (1 sample)
    }

    // Denormalize prediction
    denormalizePrediction(normalizedPred) {
        const meanY = this.normStats.mean_y[0];
        const stdY = this.normStats.std_y[0];

        return normalizedPred * stdY + meanY;
    }

    // Main prediction function
    predict(formData) {
        if (!this.parameters || !this.normStats) {
            console.error('❌ Model not initialized. Call initialize() first.');
            return null;
        }

        try {
            // Step 1: Preprocess input
            const features = this.preprocessInput(formData);
            
            // Step 2: Normalize input
            const normalizedInput = this.normalizeInput(features);
            
            // Step 3: Forward propagation
            const normalizedPrediction = this.forwardPropagation(normalizedInput);
            
            // Step 4: Denormalize output
            const actualPrice = this.denormalizePrediction(normalizedPrediction[0][0]);
            
            return actualPrice;
        } catch (error) {
            console.error('❌ Error during prediction:', error);
            return null;
        }
    }
}

// ============================================
// UI INTEGRATION CODE
// ============================================

// Create global predictor instance
let predictor = null;

document.addEventListener('DOMContentLoaded', async () => {
    // Initialize the model on page load
    predictor = new HousePricePredictor();
    await predictor.initialize();

    // Animated Starfield Canvas
    const canvas = document.getElementById('stars-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const stars = [];
        const starCount = 200;

        class Star {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.size = Math.random() * 2;
                this.speedY = Math.random() * 0.5 + 0.1;
                this.opacity = Math.random();
                this.twinkleSpeed = Math.random() * 0.02;
            }

            update() {
                this.y += this.speedY;
                if (this.y > canvas.height) {
                    this.y = 0;
                    this.x = Math.random() * canvas.width;
                }
                this.opacity += this.twinkleSpeed;
                if (this.opacity > 1 || this.opacity < 0) {
                    this.twinkleSpeed = -this.twinkleSpeed;
                }
            }

            draw() {
                ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity})`;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        for (let i = 0; i < starCount; i++) {
            stars.push(new Star());
        }

        function animateStars() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            stars.forEach(star => {
                star.update();
                star.draw();
            });
            requestAnimationFrame(animateStars);
        }

        animateStars();

        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    }

    // Form handling
    const predictionForm = document.getElementById('predictionForm');
    const predictionResultCard = document.getElementById('predictionResult');
    const predictedPriceSpan = document.getElementById('predictedPrice');
    const scrollToPredictorBtn = document.querySelector('.scroll-to-predictor');

    if (scrollToPredictorBtn) {
        scrollToPredictorBtn.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('predictor').scrollIntoView({ behavior: 'smooth' });
        });
    }

    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Collect form data (WITHOUT heating field)
            const formData = {
                area: parseFloat(document.getElementById('area').value),
                bedrooms: parseInt(document.getElementById('bedrooms').value),
                bathrooms: parseFloat(document.getElementById('bathrooms').value),
                stories: parseInt(document.getElementById('stories').value),
                mainroad: parseInt(document.getElementById('mainroad').value),
                guestroom: parseInt(document.getElementById('guestroom').value),
                basement: parseInt(document.getElementById('basement').value),
                hotwater: parseInt(document.getElementById('hotwater').value),
                ac: parseInt(document.getElementById('ac').value),
                parking: parseInt(document.getElementById('parking').value),
                furnishing: document.getElementById('furnishing').value,
                prefarea: parseInt(document.getElementById('prefarea').value)
            };

            console.log("Form Data Collected:", formData);

            try {
                // Show loading state (optional)
                if (predictionResultCard) {
                    predictionResultCard.classList.remove('visible');
                    predictionResultCard.classList.add('hidden');
                }

                // Make prediction using the neural network model
                const predictedPrice = predictor.predict(formData);

                if (predictedPrice === null) {
                    throw new Error("Prediction failed - model not initialized");
                }

                console.log("Predicted Price:", predictedPrice);

                // Format the price
                
                const formattedPrice = new Intl.NumberFormat('en-IN', { // Changed locale to 'en-IN'
                    style: 'currency',
                    currency: 'INR', // Changed currency to 'INR'
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0
                }).format(Math.abs(predictedPrice));
                

                // Display result
                if (predictedPriceSpan) {
                    predictedPriceSpan.textContent = formattedPrice;
                }
                
                if (predictionResultCard) {
                    predictionResultCard.classList.remove('hidden');
                    predictionResultCard.classList.add('visible');
                    predictionResultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }

            } catch (error) {
                console.error("Prediction failed:", error);
                alert("Failed to get prediction. Please make sure model files are loaded correctly.");
                if (predictionResultCard) {
                    predictionResultCard.classList.add('hidden');
                    predictionResultCard.classList.remove('visible');
                }
            }
        });
    }
});
