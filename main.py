import telebot
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
bot = telebot.TeleBot('')

# Словарь для отслеживания состояния пользователей
user_states = {}

# Настройки почтового сервера
SMTP_SERVER = ''
SMTP_PORT = 465
SMTP_USER = ''
SMTP_PASSWORD = ''

# Email получателя
RECIPIENT_EMAIL = ''

# Определение простой модели (тот же класс SimpleNN)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_labels)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Загружаем данные, чтобы использовать тот же vectorizer
data_path = 'dataset.txt'
data = pd.read_csv(data_path, sep=';')
text_data = data['текст']
labels = data['класс0']

# Векторизация текста
vectorizer = CountVectorizer(max_features=10000)
vectorizer.fit(text_data)

# Проверка количества классов
num_labels = labels.nunique()

# Определение и загрузка модели
input_dim = len(vectorizer.get_feature_names_out())
model = SimpleNN(input_dim, num_labels)
model.load_state_dict(torch.load(f"class_first123123.pth"))
model.eval()

# Устройство (CUDA или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    X = vectorizer.transform([text]).toarray()
    input_ids = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()

# Функция для отправки email
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, RECIPIENT_EMAIL, msg.as_string())
    except Exception as e:
        print(f"Ошибка при отправке email: {e}")

# Функция для вывода всех введенных данных и создания инлайновых кнопок
def review_data(user_id, data):
    user_data = data[str(user_id)]
    review_message = (f"Номер телефона: {user_data.get('номер телефона', 'не указано')}\n"
                      f"Почта: {user_data.get('почта', 'не указано')}\n"
                      f"Тип приложения/модели: {user_data.get('тип приложения', user_data.get('тип модели', 'не указано'))}\n"
                      f"Где используется: {user_data.get('где используется', 'не указано')}\n"
                      f"Стек: {user_data.get('стек', 'не указано')}")
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Изменить номер телефона", callback_data="edit_phone"))
    markup.add(InlineKeyboardButton("Изменить почту", callback_data="edit_email"))
    markup.add(InlineKeyboardButton("Изменить тип", callback_data="edit_type"))
    markup.add(InlineKeyboardButton("Изменить использование", callback_data="edit_usage"))
    markup.add(InlineKeyboardButton("Подтвердить", callback_data="confirm"))
    return review_message, markup

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Здравствуйте! Я Ваш личный Бот помощник, который поможет Вам составить заявку на заказ. Пожалуйста, напишите свое сообщение.")
    user_states[message.from_user.id] = 'awaiting_message'

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    user_text = message.text.lower()
    state = user_states.get(user_id, '')
    database_file = 'database.json'

    if os.path.exists(database_file):
        with open(database_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    if state == 'awaiting_message':
        data[str(user_id)] = {'сообщение': user_text, 'отзыв': ''}
        predicted_class = predict(user_text)

        if predicted_class == 0 or predicted_class == 1:
            user_states[user_id] = 'awaiting_stack'
            bot.reply_to(message, "Ваше сообщение сохранено. Теперь укажите стек.")
        elif predicted_class == 2:
            user_states[user_id] = 'awaiting_phone_2'
            bot.reply_to(message, "Ваше сообщение сохранено. Теперь укажите номер телефона.")
        else:
            user_states[user_id] = 'awaiting_review'
            bot.reply_to(message, "Ваше сообщение сохранено. Оставьте пожалуйста отзыв.")
    
    elif state == 'awaiting_stack':
        data[str(user_id)]['стек'] = user_text
        user_states[user_id] = 'awaiting_phone'
        bot.reply_to(message, "Теперь укажите номер телефона.")
    
    elif state == 'awaiting_phone':
        data[str(user_id)]['номер телефона'] = user_text
        user_states[user_id] = 'awaiting_email'
        bot.reply_to(message, "Теперь укажите почту.")
    
    elif state == 'awaiting_email':
        data[str(user_id)]['почта'] = user_text
        user_states[user_id] = 'awaiting_app_type'
        bot.reply_to(message, "Теперь укажите тип приложения.")
    
    elif state == 'awaiting_app_type':
        data[str(user_id)]['тип приложения'] = user_text
        user_states[user_id] = 'awaiting_usage'
        bot.reply_to(message, "Теперь укажите где используется.")
    
    elif state == 'awaiting_usage':
        data[str(user_id)]['где используется'] = user_text
        review_message, markup = review_data(user_id, data)
        bot.send_message(message.chat.id, review_message, reply_markup=markup)
        user_states[user_id] = 'awaiting_review'
    
    elif state == 'awaiting_phone_2':
        data[str(user_id)]['номер телефона'] = user_text
        user_states[user_id] = 'awaiting_email_2'
        bot.reply_to(message, "Теперь укажите почту.")
    
    elif state == 'awaiting_email_2':
        data[str(user_id)]['почта'] = user_text
        user_states[user_id] = 'awaiting_model_type'
        bot.reply_to(message, "Теперь укажите тип модели.")
    
    elif state == 'awaiting_model_type':
        data[str(user_id)]['тип модели'] = user_text
        user_states[user_id] = 'awaiting_model_usage'
        bot.reply_to(message, "Теперь укажите где используется.")
    
    elif state == 'awaiting_model_usage':
        data[str(user_id)]['где используется'] = user_text
        review_message, markup = review_data(user_id, data)
        bot.send_message(message.chat.id, review_message, reply_markup=markup)
        user_states[user_id] = 'awaiting_review'
    
    elif state == 'awaiting_review':
        data[str(user_id)]['отзыв'] = user_text
        user_states[user_id] = ''
        bot.reply_to(message, "Ваш отзыв сохранен. Спасибо!")
        user_data = data[str(user_id)]
        email_body = (f"Сообщение: {user_data.get('сообщение', '')}\n"
                      f"Отзыв: {user_data.get('отзыв', '')}\n"
                      f"Стек: {user_data.get('стек', '')}\n"
                      f"Номер телефона: {user_data.get('номер телефона', '')}\n"
                      f"Почта: {user_data.get('почта', '')}\n"
                      f"Тип приложения/модели: {user_data.get('тип приложения', user_data.get('тип модели', ''))}\n"
                      f"Где используется: {user_data.get('где используется', '')}")
        send_email("Новая заявка от пользователя", email_body)

    with open(database_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Обработчик инлайновых кнопок
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    user_id = call.from_user.id
    data = {}
    database_file = 'database.json'
    
    if os.path.exists(database_file):
        with open(database_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

    if call.data == "edit_phone":
        user_states[user_id] = 'awaiting_phone'
        bot.send_message(call.message.chat.id, "Введите новый номер телефона.")
    elif call.data == "edit_email":
        user_states[user_id] = 'awaiting_email'
        bot.send_message(call.message.chat.id, "Введите новую почту.")
    elif call.data == "edit_type":
        if 'тип приложения' in data[str(user_id)]:
            user_states[user_id] = 'awaiting_app_type'
            bot.send_message(call.message.chat.id, "Введите новый тип приложения.")
        else:
            user_states[user_id] = 'awaiting_model_type'
            bot.send_message(call.message.chat.id, "Введите новый тип модели.")
    elif call.data == "edit_usage":
        user_states[user_id] = 'awaiting_usage'
        bot.send_message(call.message.chat.id, "Введите новое использование.")
    elif call.data == "confirm":
        user_states[user_id] = 'awaiting_review'
        bot.send_message(call.message.chat.id, "Оставьте пожалуйста отзыв.")

# Запуск бота
bot.polling()