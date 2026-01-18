# 🇧🇩 Bangladesh Demand Prediction System

A machine learning project that predicts **clothes** and **medicine** demand across Bangladesh using population and demographic data to help retailers and pharmacies optimize inventory planning.

## 🎯 Project Goal

Predicting clothes and medicine demand using machine learning to help retailers and pharmacies optimize inventory planning across Bangladesh's major cities.

## 🌟 Features

### 📱 Web Application
- **Modern UI**: Clean, responsive web interface
- **Dual Prediction**: Clothes and medicine demand forecasting
- **Interactive Charts**: Actual vs predicted trend visualization
- **Real-time Results**: Instant predictions with accuracy metrics

### 🤖 Machine Learning Models
- **Algorithm**: Random Forest Regressor
- **Accuracy**: 85% for clothes, 79% for medicine
- **Features**: Population, density, year, and historical data
- **Validation**: Cross-validation across districts

## 📊 Predictions Available

### 👕 Clothes Items
- Shirts, Pants, Jackets, Sarees, Dresses, Coats

### 💊 Medicine Types  
- Antibiotics, Painkillers, Antacids, Vitamins, Antihistamines, Insulin

### 🏙️ Coverage Areas
- Dhaka North/South, Gazipur, Chittagong City
- Cox's Bazar, Khulna City, Rajshahi City
- Sylhet City, Rangpur City, Barisal City

## 🚀 Quick Start

### Web Application
1. Open `index.html` in your browser
2. Select **Clothes** or **Medicine** tab
3. Choose **Year**, **Item**, and **Area**
4. Click **"🔮 Predict Demand"** or **"📈 Show Trend"**

### Python Models
```bash
# Clothes prediction
cd clothes-demand-prediction
python test_and_train.py

# Medicine prediction  
cd medicine-demand-prediction
python test_and_train.py
```

## 📁 Project Structure

```
predicting-medicine-and-clothes-demand-in-Bangladesh/
├── index.html                          # Web application
├── style.css                           # Styling
├── script.js                           # JavaScript functionality
├── clothes-demand-prediction/
│   ├── bangladesh_clothes_demand.csv   # Training data
│   ├── test_and_train.py              # ML model & interface
│   └── build_clothes_demand_csv.py     # Data preprocessing
├── medicine-demand-prediction/
│   ├── bangladesh_medicine_demand.csv  # Training data
│   ├── test_and_train.py              # ML model & interface
│   └── build_medicine_demand_csv.py    # Data preprocessing
└── README.md                           # This file
```

## 🔧 Technical Details

### Dependencies
```python
pandas          # Data manipulation
numpy           # Numerical operations
scikit-learn    # Machine learning
matplotlib      # Visualization
```

### Model Performance
- **Clothes Model**: R² = 0.85, MAE = ~15 units
- **Medicine Model**: R² = 0.79, MAE = ~12 units
- **Training Time**: < 5 seconds
- **Prediction Time**: < 0.1 seconds

### Features Used
- Population size
- Population density  
- Year (scaled)
- Previous year demand (lag features)

## 📈 Model Accuracy Metrics

- **R² Score**: Percentage of variance explained (higher = better)
- **MAE**: Mean Absolute Error in units (lower = better)  
- **RMSE**: Root Mean Squared Error (lower = better)
- **Cross-validation**: Tested across all districts

## 🎨 Web Interface Features

- **Responsive Design**: Works on desktop and mobile
- **Interactive Charts**: Using Chart.js for visualizations
- **Loading Animations**: Smooth user experience
- **Error Handling**: User-friendly error messages
- **Accuracy Display**: Shows model confidence metrics

## 💡 Use Cases

### For Retailers
- **Seasonal Planning**: Predict demand for different clothing items
- **Inventory Optimization**: Avoid overstocking or stockouts
- **Regional Analysis**: Understand demand patterns by area

### For Pharmacies & Hospitals
- **Medicine Stocking**: Ensure adequate supply of essential medicines
- **Emergency Planning**: Predict demand during health crises
- **Cost Optimization**: Reduce waste from expired medicines

### For Government
- **Policy Planning**: Healthcare and textile industry insights
- **Resource Allocation**: Distribute supplies efficiently
- **Economic Forecasting**: Understand market trends

## 🔮 Future Enhancements

- **Real-time Data Integration**: Connect to live demographic data
- **More Cities**: Expand to rural areas and smaller cities
- **Seasonal Factors**: Include weather and cultural events
- **API Development**: RESTful API for third-party integration
- **Mobile App**: Native mobile application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@istiakadil14](https://github.com/IstiakAdil14/)
- Email: istiakadil346@gmail.com

## 🙏 Acknowledgments

- Bangladesh Bureau of Statistics for demographic data insights
- Scikit-learn community for machine learning tools
- Chart.js for beautiful data visualizations

---

**⭐ Star this repository if you found it helpful!**
