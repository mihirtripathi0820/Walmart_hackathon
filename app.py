
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_session import Session
import sqlite3
import os
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the dataset
data = pd.read_csv('wallmart_project/BigBasket Products.csv')
data['index'] = data.index  # Ensure there's an index column for product_id matching

# Extract categories and featured products
categories = data['Category'].unique()
featured_products = data.sample(10).to_dict(orient='records')

# One-Hot Encoder for subcategories and brands
one_hot_encoder_sub = OneHotEncoder(handle_unknown='ignore')
subcategory_matrix = one_hot_encoder_sub.fit_transform(data[['subcategory']])
one_hot_encoder_brand = OneHotEncoder(handle_unknown='ignore')
brand_matrix = one_hot_encoder_brand.fit_transform(data[['brand']])

# Impute missing values in ratings (replace with mean)
imputer = SimpleImputer(strategy='mean')
rating_matrix = imputer.fit_transform(data[['rating']])

# Standard Scaler for ratings
scaler = StandardScaler()
rating_matrix = scaler.fit_transform(rating_matrix)

# Combine all feature matrices with weights
weights = {
    'subcategory': 0.3,
    'rating': 0.5,
    'brand': 0.2
}

combined_matrix = hstack([
    subcategory_matrix * weights['subcategory'],
    rating_matrix * weights['rating'],
    brand_matrix * weights['brand']
]).toarray()


# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(combined_matrix, combined_matrix)

# Define a function to get recommendations
def get_recommendations(product_id, cosine_sim=cosine_sim):
    try:
        idx = data.index[data['index'] == product_id].tolist()[0]
    except IndexError:
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]

    return data.iloc[product_indices][['index', 'product', 'subcategory', 'brand', 'rating']]

# Database setup
def get_db_connection():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                points INTEGER NOT NULL,
                tier TEXT NOT NULL,
                avatar TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS challenges (
                name TEXT PRIMARY KEY,
                completed BOOLEAN NOT NULL,
                reward INTEGER NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS quests (
                name TEXT PRIMARY KEY,
                completed BOOLEAN NOT NULL,
                reward INTEGER NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS missions (
                name TEXT PRIMARY KEY,
                completed BOOLEAN NOT NULL,
                reward INTEGER NOT NULL
            )
        ''')
        # Insert default challenges, quests, missions
        conn.executemany('''
            INSERT OR IGNORE INTO challenges (name, completed, reward)
            VALUES (?, ?, ?)
        ''', [
            ("Purchase 5 items", False, 50),
            ("Leave 3 reviews", False, 30)
        ])
        conn.executemany('''
            INSERT OR IGNORE INTO quests (name, completed, reward)
            VALUES (?, ?, ?)
        ''', [
            ("Find hidden discount", False, 50),
            ("Explore new category", False, 30)
        ])
        conn.executemany('''
            INSERT OR IGNORE INTO missions (name, completed, reward)
            VALUES (?, ?, ?)
        ''', [
            ("Invite a friend", False, 20),
            ("Explore a new category", False, 10)
        ])

init_db()

class User(UserMixin):
    def __init__(self, user_id):
        self.user_id = user_id

    def get_id(self):
        return self.user_id

    @property
    def points(self):
        conn = get_db_connection()
        user = conn.execute('SELECT points FROM users WHERE id = ?', (self.user_id,)).fetchone()
        conn.close()
        return user['points']

    @points.setter
    def points(self, value):
        conn = get_db_connection()
        conn.execute('UPDATE users SET points = ? WHERE id = ?', (value, self.user_id))
        conn.commit()
        conn.close()
        self.update_tier()

    @property
    def tier(self):
        conn = get_db_connection()
        user = conn.execute('SELECT tier FROM users WHERE id = ?', (self.user_id,)).fetchone()
        conn.close()
        return user['tier']

    @tier.setter
    def tier(self, value):
        conn = get_db_connection()
        conn.execute('UPDATE users SET tier = ? WHERE id = ?', (value, self.user_id))
        conn.commit()
        conn.close()

    @property
    def avatar(self):
        conn = get_db_connection()
        user = conn.execute('SELECT avatar FROM users WHERE id = ?', (self.user_id,)).fetchone()
        conn.close()
        return user['avatar']

    @avatar.setter
    def avatar(self, value):
        conn = get_db_connection()
        conn.execute('UPDATE users SET avatar = ? WHERE id = ?', (value, self.user_id))
        conn.commit()
        conn.close()

    def update_points(self, amount):
        conn = get_db_connection()
        new_points = self.points + amount
        conn.execute('UPDATE users SET points = ? WHERE id = ?', (new_points, self.user_id))
        conn.commit()
        conn.close()
        self.points = new_points  # Update the local property
        self.update_tier()  # Update tier based on new points

    def update_tier(self):
        points = self.points
        if points >= 1000:
            self.tier = "Gold"
        elif points >= 500:
            self.tier = "Silver"
        else:
            self.tier = "Bronze"

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id else None

@app.route('/')
def index():
    products = data[['index', 'product', 'subcategory', 'brand', 'rating']].to_dict(orient='records')
    
    # Get the recommendations based on previously clicked products
    clicked_product_ids = session.get('clicked_products', [])
    recommendations = []
    for product_id in clicked_product_ids:
        recommendations.extend(get_recommendations(product_id).to_dict(orient='records'))
    
    # Remove duplicates
    recommendations = [dict(t) for t in {tuple(d.items()) for d in recommendations}]
    
    return render_template('index.html', categories=categories, featured_products=featured_products, products=products, recommendations=recommendations)

@app.route('/products', endpoint='show_products')
@login_required
def show_products():
    all_products = data.to_dict(orient='records')
    return render_template('products.html', products=all_products, categories=categories)

@app.route('/category/<category_name>')
@login_required
def show_category(category_name):
    category_products = data[data['Category'] == category_name].to_dict(orient='records')
    return render_template('products.html', products=category_products, category_name=category_name)

@app.route('/product/<int:product_id>')
@login_required
def product_detail(product_id):
    product = data[data['index'] == product_id].iloc[0]
    
    # Store the clicked product in session
    clicked_product_ids = session.get('clicked_products', [])
    if product_id not in clicked_product_ids:
        clicked_product_ids.append(product_id)
    session['clicked_products'] = clicked_product_ids
    
    recommendations = get_recommendations(product_id).to_dict(orient='records')
    return render_template('product_details.html', product=product, recommendations=recommendations)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ? AND password = ?', (user_id, password)).fetchone()
        conn.close()
        if user:
            login_user(User(user_id))
            return redirect(url_for('user_info'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        conn = get_db_connection()
        existing_user = conn.execute('SELECT id FROM users WHERE id = ?', (user_id,)).fetchone()
        if existing_user:
            flash('User already exists')
        else:
            conn.execute('INSERT INTO users (id, password, points, tier, avatar) VALUES (?, ?, ?, ?, ?)', (user_id, password, 0, 'Bronze', 'default_avatar.png'))
            conn.commit()
            conn.close()
            flash('Registration successful, please log in.')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/user_info')
@login_required
def user_info():
    # Fetch challenges, quests, and missions from the database
    conn = get_db_connection()
    challenges = {row['name']: row for row in conn.execute('SELECT * FROM challenges').fetchall()}
    quests = {row['name']: row for row in conn.execute('SELECT * FROM quests').fetchall()}
    missions = {row['name']: row for row in conn.execute('SELECT * FROM missions').fetchall()}
    conn.close()

    return render_template(
        'user_info.html',
        user=current_user,
        challenges=challenges,
        quests=quests,
        missions=missions
    )

@app.route('/search')
@login_required
def search():
    query = request.args.get('query', '')
    category = request.args.get('category', '')
    
    # Filter products based on query and category
    if category:
        filtered_data = data[data['Category'] == category]
    else:
        filtered_data = data
    
    if query:
        filtered_data = filtered_data[
            filtered_data['product'].str.contains(query, case=False, na=False) |
            filtered_data['subcategory'].str.contains(query, case=False, na=False) |
            filtered_data['brand'].str.contains(query, case=False, na=False)
        ]
    
    filtered_products = filtered_data.to_dict(orient='records')
    return render_template('products.html', products=filtered_products, categories=categories)

