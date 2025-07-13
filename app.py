from flask import Flask, render_template, request, redirect, url_for
from forms import LoginForm
import pandas as pd
from customer_segmentation_recommendation_system import recommendations_df  # Importing the recommendations data


if 'cluster' in recommendations_df.columns:
    recommendations_df2 = recommendations_df.drop('cluster', axis=1)
print(recommendations_df2.head())

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Sample data frame structure
'''
data = {
    'CustomerID': [12347, 12348, 12349],
    'cluster': [0, 1, 2],
    'Rec1_StockCode': ['85123A', '85099B', '47566'],
    'Rec1_Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'JUMBO BAG RED RETROSPOT', 'PARTY BUNTING'],
    'Rec2_StockCode': ['22423', '20725', '22720'],
    'Rec2_Description': ['REGENCY CAKESTAND 3 TIER', 'LUNCH BAG RED RETROSPOT', 'SET OF 3 CAKE TINS PANTRY DESIGN'],
    'Rec3_StockCode': ['47566', '20725', '21731'],
    'Rec3_Description': ['PARTY BUNTING', 'LUNCH BAG RED RETROSPOT', 'RED TOADSTOOL LED NIGHT LIGHT']
}
'''
# Convert the data into a pandas DataFrame
#recommendations_df = pd.DataFrame(recommendations_dict)
# Ensure the CustomerID is the index for easier lookup
#recommendations_df.set_index('CustomerID', inplace=True)

@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Convert input to integer
        customer_id = int(float(form.customer_id.data))
        # Redirect to the product page with the customer ID
        return redirect(url_for('products', customer_id=customer_id))
    return render_template('login.html', form=form)

@app.route('/products/<int:customer_id>')
def products(customer_id):
    try:
        # Fetch the recommendations for the given customer ID
        customer_recommendations = recommendations_df2.loc[customer_id]
        # If the result is a Series, convert it to a DataFrame
        if isinstance(customer_recommendations, pd.Series):
            customer_recommendations = customer_recommendations.to_frame().transpose()
        # Filter the DataFrame to only include the recommendation columns
        products = customer_recommendations[['Rec1_StockCode', 'Rec1_Description', 'Rec2_StockCode', 'Rec2_Description', 'Rec3_StockCode', 'Rec3_Description']]
    except KeyError:
        # Handle case where customer ID is not found
        products = pd.DataFrame({
            'Rec1_StockCode': ['Not found'],
            'Rec1_Description': ['No product found'],
            'Rec2_StockCode': ['Not found'],
            'Rec2_Description': ['No product found'],
            'Rec3_StockCode': ['Not found'],
            'Rec3_Description': ['No product found']
        })

    # Convert products to a list of dictionaries for easy rendering
    products_list = products.to_dict('records')
    return render_template('products.html', products=products_list, customer_id=customer_id)

if __name__ == '__main__':
    app.run(debug=True)