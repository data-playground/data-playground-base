import os
import datetime
from flask import Flask, Blueprint, render_template, redirect, url_for, flash, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, current_user, login_required
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, HiddenField, SelectField, IntegerField, TextAreaField 
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
import requests # Import requests for API calls
from sqlalchemy.types import TypeDecorator, String
from sqlalchemy import Unicode
import json
import re
import ast
from markupsafe import Markup
from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token
from dotenv import load_dotenv

load_dotenv()

# --- Configuration for Barcode Lookup API ---
BARCODE_API_URL = "https://api.barcodelookup.com/v3/products"
OPENLIBRARY_URLS = {
    'isbn': "https://openlibrary.org/isbn/",
    'works': "https://openlibrary.org/works/",
    'authors': "https://openlibrary.org/authors/",
}
# IMPORTANT: Use a secure method for your API key. For demonstration, we use a placeholder.
# Replace 'YOUR_PLACEHOLDER_API_KEY' with your actual API key.
BARCODE_LOOKUP_API_KEY = os.environ.get('BARCODE_LOOKUP_API_KEY', os.getenv('BARCODE_LOOKUP_API_KEY'))


class JsonList(TypeDecorator):
    """
    Custom type for storing Python objects (e.g., list of dicts, list of strings) 
    as a JSON string in the database.
    It handles serialization on the way in and deserialization on the way out, 
    including handling Python-style single quotes in input.
    """
    impl = Unicode # Use Unicode/String to store the JSON string

    def process_bind_param(self, value, dialect):
        """
        Converts Python object (list/dict/string) input to JSON string for saving.
        Attempts to fix Python-style single quotes if JSON loading fails.
        """
        if value is None or value == '':
            return None
        
        # If input is a string (from a form field)
        if isinstance(value, str):
            data = None
            try:
                # 1. Attempt to load as pure JSON
                data = json.loads(value)
            except json.JSONDecodeError:
                # 2. If it fails, attempt to safely evaluate as a Python literal string 
                # (This handles Python's use of single quotes for list/dict delimiters)
                try:
                    data = ast.literal_eval(value) 
                except (ValueError, SyntaxError):
                    # 3. Fallback: treat as comma-separated list of strings
                    value = [item.strip() for item in value.split(',') if item.strip()]
                    return json.dumps(value)

            # If loading succeeded, ensure the result is a list or dict
            if isinstance(data, (list, dict)): 
                value = data
            elif data is not None:
                # If it parsed but wasn't a structure (e.g., just a single string), wrap it.
                value = [data]
        
        # Finally, dump the resulting structure to a JSON string for storage
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        """
        Converts JSON string from DB back to Python object (list of dicts, etc.) upon retrieval.
        Adds robust handling for mixed quotes, apostrophes, and double-encoding.
        """
        if value is None:
            return []

        # Function to try loading, handling one level of double-encoding and quote cleanup
        def robust_load(data_str):
            if not data_str:
                return None
                
            # Attempt 1: Load the value directly as JSON
            try:
                result = json.loads(data_str)
                
                # Handle double encoding: if result is a string, try loading it one more time
                if isinstance(result, str):
                    try:
                        # Attempt to load the inner string (which may still contain apostrophes)
                        inner_result = json.loads(result)
                        return inner_result
                    except (json.JSONDecodeError, TypeError):
                        # If the inner string isn't valid JSON, return the outer result (string)
                        return result 
                
                # If result is already a list or dict, we succeeded
                return result

            except (json.JSONDecodeError, TypeError):
                # Attempt 2: Failed as pure JSON. Try to evaluate as a Python literal.
                # This handles strings that use single quotes as list/dict delimiters, 
                # but correctly maintains apostrophes inside string content.
                try:
                    return ast.literal_eval(data_str)
                except (ValueError, SyntaxError, NameError):
                    return None # Failed to evaluate as a safe literal

        # 1. Try robust loading logic
        result = robust_load(value)
        if isinstance(result, (list, dict)):
            return result
        
        # Fallback: Treat the raw string as a simple comma-separated list of strings
        # This is for corrupted or manually entered non-JSON data.
        return [item.strip() for item in value.split(',') if item.strip()]
        
        
# --- Application and Extension Setup ---
app = Flask(__name__)

# Basic Configuration
# Use a secret key that is difficult to guess
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.getenv('SECRET_KEY'))
# Configure SQLite database (database.db will be created in the instance folder)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


# --- Template Context Processor ---
@app.context_processor
def inject_global_data():
    """Injects global data like the current year into all templates."""
    return dict(current_year=datetime.datetime.now().year)


# --- User Loader for Flask-Login ---
@login_manager.user_loader
def load_user(user_id):
    """Callback function to reload the user object from the user ID stored in the session."""
    return db.session.get(User, int(user_id))

# --- Database Models (Schema) ---

class User(db.Model, UserMixin):
    """User Model: Contains authentication and profile data."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    
    # Store the unique Google ID (sub)
    google_id = db.Column(db.String(100), unique=True, nullable=True) 
    
    # Keep the password_hash for users who sign up traditionally
    password_hash = db.Column(db.String(60), nullable=True)

    # Relationships
    books_owned = db.relationship('BookCopy', backref='owner', foreign_keys='BookCopy.owner_id', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"
        
class Author(db.Model):
    """Book Model: Represents a single book in the library."""
    id = db.Column(db.Integer, primary_key=True)
    op_id = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    remote_ids = db.Column(JsonList(), nullable=True)
    
    # Relationships
    books = db.relationship('Book', backref='author_obj', lazy=True)

    def __repr__(self):
        return f"Author('{self.name}')"
        
class TempTable(db.Model):
    """TempTable Model: Used to save data that will be needed later in the session. Data for session is cleaned after."""
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(JsonList(), nullable=True)
    
    def __repr__(self):
        return f"Record {self.id} saved"

class Book(db.Model):
    """Book Model: Represents a single book in the library."""
    __tablename__ = 'book'
    id = db.Column(db.Integer, primary_key=True)
    op_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(100), nullable=False)
    author = db.Column(db.Integer, db.ForeignKey('author.id'), nullable=False)
    release_date = db.Column(db.String(100), nullable=True)
    description = db.Column(db.String(1000), nullable=True)
    features = db.Column(JsonList(), nullable=True)
    contributors = db.Column(JsonList(), nullable=True)
    links = db.Column(JsonList(), nullable=True)
    
    # Relationships
    copies = db.relationship('BookCopy', backref='book', lazy=True)
    comments = db.relationship('Comment', backref='book', lazy=True, order_by="Comment.timestamp.desc()")

    def __repr__(self):
        return f"Book('{self.title}', Author ID: '{self.author_id}')"
        
class BookCopy(db.Model):
    __tablename__ = 'bookcopy'
    id = db.Column(db.Integer, primary_key=True)
    op_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(100), nullable=False)
    num_pages = db.Column(db.Integer, nullable=True)
    isbn_10 = db.Column(db.String(20), nullable=True) # Unique identifier for a specific edition
    isbn_13 = db.Column(db.String(20), nullable=True) # Unique identifier for a specific edition
    edition_date = db.Column(db.String(100), nullable=True)
    thumbnail = db.Column(JsonList(), nullable=True)
    classifications = db.Column(JsonList(), nullable=True)
    contributors = db.Column(JsonList(), nullable=True)
    publishers = db.Column(JsonList(), nullable=True)
    identifiers = db.Column(JsonList(), nullable=True)
    series = db.Column(JsonList(), nullable=True)
        
    # Status: 'Available', 'Lent', 'Lost'
    status = db.Column(db.String(20), nullable=False, default='Available')

    # Foreign Keys
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False) # Link to conceptual book
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    current_lender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    # Relationships
    current_lender = db.relationship('User', backref='copies_borrowed', foreign_keys=[current_lender_id], lazy=True)
    loans = db.relationship('Loan', backref='bookcopy', lazy=True)
    # book = db.relationship('Book', backref='bookcopy', lazy=True)


    def is_available(self):
        return self.status == 'Available'

    def __repr__(self):
        return f"BookCopy('{self.book_id}, ISBN: {self.isbn}, Status: '{self.status}')"


class Loan(db.Model):
    """Loan Model: Tracks the history of book borrowings."""
    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey('bookcopy.id'), nullable=False)
    borrower_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    loan_date = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    return_date = db.Column(db.DateTime, nullable=True)

    # Relationships
    borrower = db.relationship('User', backref='loans_history', foreign_keys=[borrower_id], lazy=True)

    def __repr__(self):
        return f"Loan(Book ID: {self.book_id}, Borrower ID: {self.borrower_id}, Lent: {self.loan_date}, Returned: {self.return_date})"


class Comment(db.Model):
    """Comment Model: User feedback on a specific book."""
    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"Comment(Book ID: {self.book_id}, User ID: {self.user_id}, Time: {self.timestamp})"


# --- Forms ---

class RegistrationForm(FlaskForm):
    """Form for new user registration."""
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = db.session.execute(db.select(User).filter_by(username=username.data)).scalar_one_or_none()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = db.session.execute(db.select(User).filter_by(email=email.data)).scalar_one_or_none()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')

class LoginForm(FlaskForm):
    """Form for user login."""
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class BarcodeLookupForm(FlaskForm):
    """NEW: Simple form to handle only the barcode lookup."""
    barcode = StringField('Barcode/ISBN', validators=[DataRequired(), Length(max=50)])
    lookup_submit = SubmitField("Fetch Details")

class AddBookForm(FlaskForm):
    """
    MODIFIED: Form to add a new book. Only contains the required manual fields.
    DataRequired() is now safe here because fields are pre-populated after lookup.
    """
    title = StringField('Title', validators=[DataRequired(), Length(min=1, max=100)])
    author = StringField('Author', validators=[DataRequired(), Length(min=1, max=100)])
    edition_date = StringField('Release Date')
    description = TextAreaField('Description')
    publishers = StringField('Publishers')
    features = StringField('Features')
    thumbnail = StringField('Thumbnail')
    
    # Hidden field to preserve the barcode data for display/logging if needed, but not for API lookup
    persisted_barcode = HiddenField() 
    
    add_submit = SubmitField('Add Book') # Used for final save

class CommentForm(FlaskForm):
    """Form to add a comment to a book."""
    text = TextAreaField('Your Comment', validators=[DataRequired()])
    submit = SubmitField('Post Comment')

# --- Routes (Endpoints) ---

@app.route("/")
@app.route("/home")
def home():
    """Homepage: Displays all available books for lending."""
    # Fetch all books that are currently 'Available'
    all_books = db.session.execute(db.select(Book).order_by(Book.title.asc())).scalars().all()
    
    book_summaries = []
    
    for book in all_books:
        total_copies = len(book.copies)
        available_copies = sum(1 for copy in book.copies if copy.status == 'Available')
        
        book_summaries.append({
            'book': book,
            'total_copies': total_copies,
            'available_copies': available_copies
        })
    
    return render_template('index.html', title='Available Books', summaries=book_summaries, is_search=False)

@app.route("/search")
def search():
    """Search functionality."""
    query_text = request.args.get('q', '').strip()
    if query_text:
        # Search books by title or author (case-insensitive)
        search_pattern = f"%{query_text}%"
        books = db.session.execute(
            db.select(Book).filter(
                (Book.title.ilike(search_pattern)) | 
                (Book.author.ilike(search_pattern))
            )
        ).scalars().all()
        
        book_summaries = []
        
        for book in books:
            total_copies = len(book.copies)
            available_copies = sum(1 for copy in book.copies if copy.status == 'Available')
            
            book_summaries.append({
                'book': book,
                'total_copies': total_copies,
                'available_copies': available_copies
            })
        
        flash(f"Found {len(books)} results for '{query_text}'", 'success')
        return render_template('index.html', title=f'Search Results for "{query_text}"', summaries=book_summaries, is_search=True)
    else:
        # If no query, redirect to home
        return redirect(url_for('home'))

@app.route("/register", methods=['GET', 'POST'])
def register():
    """User registration route."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        # Hash the password and create the user
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth.html', title='Register', form=form, auth_type='register')

# @app.route("/login", methods=['GET', 'POST'])
# def login():
    # """User login route."""
    # if current_user.is_authenticated:
        # return redirect(url_for('home'))
    # form = LoginForm()
    # if form.validate_on_submit():
        # user = db.session.execute(db.select(User).filter_by(email=form.email.data)).scalar_one_or_none()
        # if user and bcrypt.check_password_hash(user.password_hash, form.password.data):
            # # Log the user in and remember their session
            # login_user(user)
            # next_page = request.args.get('next')
            # return redirect(next_page) if next_page else redirect(url_for('home'))
        # else:
            # flash('Login Unsuccessful. Please check email and password', 'danger')
    # return render_template('auth.html', title='Login', form=form, auth_type='login')

oauth = OAuth(app)

# Register Google OAuth service
# Make sure to set environment variables: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    # Retrieve configuration dynamically
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'} # Request basic user info
)

# auth = Blueprint('auth', __name__)

@app.route('/login')
def login():
    """Renders the login page with the Google button."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index')) # Redirect if already logged in
    return render_template('login.html')

# ----------------------------------------------------------------------

@app.route('/google/login')
def google_login():
    """Initiates the Google OAuth 2.0 flow."""
    
    # 1. Generate and store the nonce in the session
    nonce = generate_token()
    session['nonce'] = nonce
    
    # Redirect user to Google's authorization endpoint
    redirect_uri = url_for('google_auth', _external=True)
    
    # 2. Pass the nonce to Google in the authorization request
    return oauth.google.authorize_redirect(redirect_uri, nonce=nonce)

# ----------------------------------------------------------------------

@app.route('/google/auth')
def google_auth():
    """Handles the callback from Google, exchanges the code for tokens."""
    # --- Step 1: Exchange the authorization code for an ID Token ---
    try:
        # This step handles state verification, token exchange, and ID Token validation
        token = oauth.google.authorize_access_token()
    except Exception as e:
        # Handle errors (e.g., user denied access, invalid state)
        print(f"OAuth Error: {e}")
        return redirect(url_for('login'))


    # --- Step 2: Retrieve and Validate the Nonce ---
    
    # SECURITY: The nonce is popped from the session. 
    # It must be retrieved from the session because it was stored there 
    # just before the redirect to Google.
    nonce = session.pop('nonce', None)
    
    if not nonce:
        # If the nonce is missing, the session may have expired or there 
        # is a potential security issue.
        # Log this critical error.
        return "Nonce missing. Session expired or security error.", 400

    # --- Step 3: Parse and Validate the ID Token ---
    
    # FIX: Pass the retrieved nonce to the parse_id_token() method
    try:
        user_info = oauth.google.parse_id_token(token, nonce=nonce) 
    except Exception as e:
        print(f"ID Token Validation Error: {e}")
        # If token validation (including nonce match) fails, deny access.
        return redirect(url_for('login'))


    # --- Step 4: User Lookup and Session Management ---
    google_id = user_info.get('sub')
    email = user_info.get('email')
    username = user_info.get('name') # or user_info.get('given_name')
    
    user = db.session.execute(db.select(User).filter_by(google_id=google_id)).scalar_one_or_none()
    
    if user is None:
        # A new user is signing up
        # Check if the email is already in use by a traditional account
        existing_email_user = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()
        
        if existing_email_user:
            # OPTION 1: Link the existing account (recommended for user experience)
            existing_email_user.google_id = google_id
            user = existing_email_user
        else:
            # OPTION 2: Create a brand new user account
            # Note: For security, you might want a placeholder password_hash or a dedicated column to mark it as a Google-only account.
            new_user = User(
                username=username,
                email=email,
                google_id=google_id,
                password_hash=None # No local password
            )
            db.session.add(new_user)
            user = new_user
    
    db.session.commit()
    
    # 4. Log the user into Flask-Login
    login_user(user)
    
    # next_page = request.args.get('next')
    # return redirect(next_page) if next_page else redirect(url_for('home'))
    return redirect(url_for('home'))

@app.route("/logout")
@login_required
def logout():
    """User logout route."""
    logout_user()
    return redirect(url_for('home'))

@app.route("/add_book", methods=['GET', 'POST'])
@login_required
def add_book():
    """
    Allows authenticated users to add a new book.
    Form data can be pre-filled via query parameters after a successful barcode lookup.
    """
    add_form = AddBookForm()
    lookup_form = BarcodeLookupForm() # Pass this form to the template for the lookup section
    thumbnail = ''

    # If this is a GET request and we have query parameters (from a successful lookup), pre-populate the form
    if request.method == 'GET':
        barcode = request.args.get('barcode')
        
        if barcode:
            session_key = f'book_data_{barcode}'
            # Retrieve and remove the data from the session
            book_data_start = db.session.execute(
                db.select(TempTable).filter_by(id=session.get(session_key))
            ).scalars().all() 
            
            # book_data = {
                # 'authors': session.get(f'book_data_authors_{barcode}'),
                # 'book': session.get(f'book_data_book_{barcode}'),
                # 'bookcopy': session.get(f'book_data_bookcopy_{barcode}') 
            # }
                        
            if book_data_start: 
                book_data = book_data_start[0].data
                                
                add_form.title.data = book_data['book']['title']
                add_form.author.data = book_data['authors'][0]['name'] if "authors" in book_data.keys() else "N/A"
                add_form.edition_date.data = book_data['bookcopy']['edition_date']
                add_form.description.data = book_data['book']['description']
                add_form.publishers.data = book_data['bookcopy']['publishers']
                add_form.features.data = book_data['book']['features']
                add_form.thumbnail.data = book_data['bookcopy']['thumbnail']
                
                thumbnail = book_data['bookcopy']['thumbnail']
            
                # Pass the barcode to a hidden field for potential display/logging
                add_form.persisted_barcode.data = barcode 
                
                flash(f'''Book details loaded from barcode lookup: "{book_data['book']['title']}" by {book_data['book']['author']}. Please confirm and add a copy.''', 'info')
                
            else:
                # FAILURE: Barcode was in URL, but data was NOT in session (e.g., failed lookup or session expiration)
                add_form.persisted_barcode.data = barcode
                flash(f"Could not retrieve data for barcode {barcode}. Please enter details manually or retry the lookup.", 'warning')

    # Handle final submission of the AddBookForm
    if add_form.validate_on_submit() and add_form.add_submit.data:
        barcode = request.args.get('barcode')
                
        if barcode:
            session_key = f'book_data_{barcode}'
            # Retrieve and remove the data from the session
            book_data_start = db.session.execute(
                db.select(TempTable).filter_by(id=session.get(session_key))
            ).scalars().all()
            
            print("Extracted from session cookie")
            
            if book_data_start:
                book_data = book_data_start[0].data
        
                author_name = add_form.author.data.strip()
                book_title = add_form.title.data.strip()
                
                author = None

                if author_name and author_name != "N/A":
                    # 1. FIND OR CREATE AUTHOR
                    author = db.session.execute(db.select(Author).filter_by(op_id=book_data['authors'][0]['op_id'])).scalar_one_or_none()
                    if not author:
                        new_author = book_data['authors'][0]
                        author = Author(
                            op_id=new_author['op_id'], 
                            name=new_author['name'], 
                            # remote_ids=new_author['remote_ids']
                        )
                        db.session.add(author)
                        db.session.commit() 

                # 2. Check if the Conceptual Book already exists (by Title and Author ID)
                conceptual_book = db.session.execute(db.select(Book).filter_by(title=book_data['book']['title'], author=author.id if author else "n/A")).scalar_one_or_none()

                # If it doesn't exist, create it
                if not conceptual_book:
                    conceptual_book = Book(
                        op_id = book_data['book']['op_id'],
                        title = book_data['book']['title'],
                        author = author.id if author_name and author_name != "N/A" else "N/A", 
                        release_date = book_data['book']['release_date'],
                        description = book_data['book']['description'],
                        features = book_data['book']['features'],
                        contributors = book_data['book']['contributors'],
                        # links = book_data['book']['links']
                    )
                    db.session.add(conceptual_book)
                    db.session.commit()
                
                # 3. Create the Physical Copy linked to the Conceptual Book
                new_copy = BookCopy(
                    op_id = book_data['bookcopy']['op_id'],
                    title = book_data['bookcopy']['title'], 
                    num_pages = book_data['bookcopy']['num_pages'], 
                    isbn_10 = book_data['bookcopy']['isbn_10'], 
                    isbn_13 = book_data['bookcopy']['isbn_13'],
                    edition_date = book_data['bookcopy']['edition_date'],
                    thumbnail = book_data['bookcopy']['thumbnail'],
                    classifications = book_data['bookcopy']['classifications'],
                    contributors = book_data['bookcopy']['contributors'],
                    publishers = book_data['bookcopy']['publishers'],
                    identifiers = book_data['bookcopy']['identifiers'],
                    series = book_data['bookcopy']['series'],
                    book_id = conceptual_book.id,
                    owner_id=current_user.id,
                    status='Available'
                )
                db.session.add(new_copy)
                db.session.commit()
                
                flash(f'A new copy of "{conceptual_book.title}" (ISBN: {new_copy.isbn_13 or "N/A"}) has been added to the library!', 'success')
                return redirect(url_for('book_detail', book_id=conceptual_book.id))

    return render_template(
        'add_book.html', 
        title='Contribute Book', 
        add_form=add_form,
        lookup_form=lookup_form, # Pass both forms to the template
        thumbnail=thumbnail
    )

@app.route("/barcode_lookup", methods=['POST'])
@login_required
def barcode_lookup():
    """NEW: Dedicated route to handle the barcode lookup API call."""
    ## Old API
    # https://openlibrary.org/api/books?bibkeys=ISBN:9781501111112&jscmd=details&format=json
    
    
    # Push lookup results (entire book_data dictionary) to a temp table
    # Save ID of the record on this temp table to session cookie
    # add_book() would look for session cookie, get ID, then query temp table to get full results
    lookup_form = BarcodeLookupForm()

    if lookup_form.validate_on_submit():
        barcode = lookup_form.barcode.data.strip()
        
        try:
            # Make the API call
            response = requests.get(OPENLIBRARY_URLS['isbn'] + f'{barcode}.json', timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            data = response.json()
            
            book_data = {}
            
            if data:
                
                book_data['bookcopy'] = {
                    'op_id': re.search(r'https:\/\/openlibrary\.org\/books\/([A-Z0-9]*)\.json', response.url).group(1),
                    'title': data['title'], 
                    'isbn_10': data.get('isbn_10', [None])[0], 
                    'isbn_13': data.get('isbn_13', [None])[0],
                    'num_pages': data.get('number_of_pages'),
                    'edition_date': data.get('created')['value'],
                    'thumbnail': data.get('covers', []),
                    'classifications': data.get('lc_classifications', []),
                    'contributors': data.get('authors', []),
                    'publishers': data.get('publishers', []),
                    'identifiers': data.get('identifiers', []),
                    'series': data.get('series', []),
                    'works': data['works'][0]['key'].split('/')[-1]
                }
                                                
                if book_data['bookcopy']['works']: 
                    response = requests.get(OPENLIBRARY_URLS['works'] + f'{book_data["bookcopy"]["works"]}.json', timeout=10)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    
                    data = response.json()
                                        
                    if data:
                        description_start = data.get('description', None)
                                      
                        book_data['book'] = {
                            'op_id': book_data['bookcopy']['works'],
                            'title': data['title'],
                            'author': data['authors'][0]['author']['key'].split("/")[-1] if 'authors' in data.keys() and len(data['authors']) > 0 else book_data['bookcopy']['contributors'][0]['key'].split("/")[-1] if len(book_data['bookcopy']['contributors']) > 0 else "N/A", 
                            'release_date': data['created']['value'],
                            'description': description_start['value'] if isinstance(description_start, dict) else description_start if description_start else "N/A",
                            'features': data.get('subjects', [])[:10], # data['subjects'] + data['subject_places'] + data['subject_people'] + data['subject_times'],
                            'contributors': data.get('authors', [])
                            # 'links': data['links']
                        }
                                        
                if book_data['bookcopy']['contributors'] or book_data['book']['contributors']:
                    book_data['authors'] = []
                    authors_lst = list(set([author["key"].split("/")[-1] for author in book_data['bookcopy']['contributors']] + [author['author']["key"].split("/")[-1] for author in book_data['book']['contributors']]))
                    for author in authors_lst:
                        response = requests.get(OPENLIBRARY_URLS['authors'] + f'{author}.json', timeout=10)
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                        
                        data = response.json()
                        
                        if data:
                    
                            book_data['authors'].append({
                                'op_id': author,
                                'name': data['name'],
                                # 'remote_ids': data['remote_ids']
                            })        
                            
                title = book_data['bookcopy']['title']
                        
                if title:
                    # session[f'book_data_{barcode}'] = book_data
                    session_data = TempTable(
                        data = book_data
                    )
                    db.session.add(session_data)
                    db.session.commit()         
                    session[f'book_data_{barcode}'] = session_data.id
                    # session[f'book_data_bookcopy_{barcode}'] = book_data['bookcopy']
                    # session[f'book_data_book_{barcode}'] = book_data['book']
                    # session[f'book_data_authors_{barcode}'] = book_data['authors']
                    # Success: Redirect back to /add_book with data in query parameters
                    return redirect(url_for('add_book', barcode=barcode))
                else:
                    # Data was incomplete, redirect back to /add_book with a warning
                    flash("Barcode lookup succeeded but key book details (Title/Author) were missing. Please enter details manually.", 'warning')

            else:
                flash(f"Barcode lookup failed: No product found for barcode {barcode}.", 'warning')
        
        except requests.exceptions.Timeout:
            flash("Barcode lookup failed: Request timed out. Please try manually.", 'danger')
        except requests.exceptions.RequestException:
            flash(f"Barcode lookup failed due to network error. Please try manually.", 'danger')
        except Exception as e:
            flash(f"An unexpected error occurred during lookup. Please try manually. {e}", 'danger')
    
    # If validation failed or lookup failed, redirect back to the add_book page
    # Passing the barcode might be useful if the API failed but validation succeeded
    return redirect(url_for('add_book', barcode=lookup_form.barcode.data.strip()))


# --- NEW ROUTE: Conceptual Book Detail Page ---
@app.route("/book/<int:book_id>")
@login_required
def book_detail(book_id):
    """Displays conceptual book details and lists all physical copies."""
    book = db.session.get(Book, book_id)
    if not book:
        flash('Book title not found.', 'danger')
        return redirect(url_for('home'))

    # Fetch all physical copies (available, lent, lost) associated with this conceptual book.
    # Order by status to show available first.
    copies = db.session.execute(
        db.select(BookCopy).filter_by(book_id=book_id).order_by(BookCopy.status.asc(), BookCopy.id.asc())
    ).scalars().all()

    author_name = book.author_obj.name if book.author_obj else "Unknown Author"
    
    comment_form = CommentForm()
    
    # Reusing 'detail.html' for the main book page. The template must iterate over 'copies'.
    return render_template(
        'detail.html', 
        title=book.title,
        book=book,
        author_name=author_name,
        copies=copies,
        comment_form=comment_form,
        # copy, comment_form, active_loan are not needed on this conceptual page
    )

# --- NEW ROUTE: Dedicated Physical Copy Detail Page ---
@app.route("/copy/<int:copy_id>")
@login_required
def copy_detail(copy_id):
    """Displays specific physical copy details, loan status, and comments."""
    copy = db.session.get(BookCopy, copy_id)
    if not copy:
        flash('Physical copy not found.', 'danger')
        return redirect(url_for('home'))

    comment_form = CommentForm()
    active_loan = None
    if copy.status == 'Lent':
        # Fetch the current active loan for display purposes
        active_loan = db.session.execute(
            db.select(Loan).filter_by(
                book_id=copy_id, return_date=None
            ).order_by(Loan.loan_date.desc())
        ).scalar_one_or_none()
    
    # We pass the copy's ConceptualBook object as 'book'
    book = copy.book
    author_name = book.author_obj.name if book.author_obj else "Unknown Author"

    # Assume a new template 'copy_detail.html' for this dedicated page.
    return render_template(
        'copy_detail.html', 
        title=f"{book.title} (Copy #{copy.id})",
        book=book, # Conceptual book details
        copy=copy, # Specific physical copy details
        author_name=author_name, 
        comment_form=comment_form,
        active_loan=active_loan
    )

@app.route("/book/<int:book_id>/lend", methods=['POST'])
@login_required
def lend_book(book_id):
    """Endpoint to lend a book to the current user."""
    copy = db.session.get(BookCopy, book_id)
    if not copy or not copy.is_available():
        flash('This book copy is currently unavailable for lending.', 'danger')
        return redirect(url_for('book_detail', book_id=copy.book_id))

    # 1. Update Book status
    copy.status = 'Lent'
    copy.current_lender_id = current_user.id
    
    # 2. Create new Loan record
    loan = Loan(
        book_id=book_id,
        borrower_id=current_user.id,
        loan_date=datetime.datetime.utcnow()
    )
    
    db.session.add(loan)
    db.session.commit()
    flash(f'You have successfully borrowed "{copy.title}"!', 'success')
    return redirect(url_for('book_detail', book_id=copy.book_id))


@app.route("/book/<int:book_id>/return", methods=['POST'])
@login_required
def return_book(book_id):
    """Endpoint to return a book."""
    copy = db.session.get(BookCopy, book_id)
    
    # Check ownership/lender status for security
    is_owner_or_lender = copy.owner_id == current_user.id or copy.current_lender_id == current_user.id

    if not copy or copy.status != 'Lent' or not is_owner_or_lender:
        flash('This book copy is not currently lent out or you do not have permission to return it.', 'danger')
        return redirect(url_for('book_detail', book_id=copy.book_id))

    # 1. Update Book status
    copy.status = 'Available'
    copy.current_lender_id = None
    
    # 2. Update the active Loan record
    active_loan = db.session.execute(
        db.select(Loan).filter_by(
            book_id=book_id, 
            borrower_id=current_user.id,
            return_date=None
        )
    ).scalar_one_or_none()

    if active_loan:
        active_loan.return_date = datetime.datetime.utcnow()
    
    db.session.commit()
    flash(f'Thank you! The book "{copy.title}" has been marked as returned and is now available.', 'success')
    return redirect(url_for('book_detail', book_id=copy.book_id))


@app.route("/book/<int:book_id>/comment", methods=['POST'])
@login_required
def post_comment(book_id):
    """Endpoint to post a comment on a book."""
    book = db.session.get(Book, book_id)
    if not book:
        flash('Book not found.', 'danger')
        return redirect(url_for('home'))
    
    form = CommentForm()
    if form.validate_on_submit():
        comment = Comment(
            text=form.text.data,
            book_id=book_id,
            user_id=current_user.id,
            timestamp=datetime.datetime.utcnow()
        )
        db.session.add(comment)
        db.session.commit()
        flash('Your comment has been posted!', 'success')
    else:
        # If form validation fails (e.g., empty comment), display error on the detail page
        flash('Comment could not be posted. Please ensure the field is not empty.', 'danger')

    return redirect(url_for('book_detail', book_id=book_id))


# --- Database Initialization ---
# This block should be executed once to set up the database structure
with app.app_context():
    db.create_all()


if __name__ == '__main__':
    # Flask runs in debug mode for development.
    # Replace with a proper WSGI server (like Gunicorn) for deployment.
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000) # Listen on all interfaces on port 5000
