
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters
)
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import torch
from utils.u2net_hair import segment_hair

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def apply_lipstick(image, landmarks, color):
    h, w = image.shape[:2]
    indices = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415]
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], color)
    mask = cv2.GaussianBlur(mask, (7, 7), 10)
    return cv2.addWeighted(image, 1, mask, 0.4, 0)

def apply_hair_color(image, color):
    mask = segment_hair(image)
    color_layer = np.full_like(image, color)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    colored = np.where(mask_3d > 0.2, color_layer, image)
    return cv2.addWeighted(image, 0.6, colored, 0.4, 0)

def get_service_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💄 تغییر رنگ لب", callback_data='service_lip')],
        [InlineKeyboardButton("💇 تغییر رنگ مو", callback_data='service_hair')]
    ])

def get_color_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("قرمز", callback_data='red')],
        [InlineKeyboardButton("صورتی", callback_data='pink')],
        [InlineKeyboardButton("بنفش", callback_data='purple')],
        [InlineKeyboardButton("نارنجی", callback_data='orange')]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! کدام تغییر زیبایی را می‌خواهی؟", reply_markup=get_service_keyboard())

async def service_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    context.user_data['service'] = q.data.split('_')[1]
    await q.edit_message_text("رنگ مورد نظر را انتخاب کن:", reply_markup=get_color_keyboard())

async def color_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    cmap = {
        'red': (0, 0, 255), 'pink': (203, 192, 255),
        'purple': (128, 0, 128), 'orange': (0, 165, 255)
    }
    context.user_data['color'] = cmap.get(q.data, (0, 0, 255))
    await q.edit_message_text("حالا لطفاً یک عکس واضح از چهره‌ات بفرست.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    service = context.user_data.get('service')
    color = context.user_data.get('color', (0, 0, 255))

    file = await update.message.photo[-1].get_file()
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tf:
        await file.download_to_drive(tf.name)
        img = cv2.imread(tf.name)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            if service == 'lip':
                out = apply_lipstick(img, landmarks, color)
            elif service == 'hair':
                out = apply_hair_color(img, color)
            else:
                out = img
            out_path = tf.name.replace(".jpg", "_out.jpg")
            cv2.imwrite(out_path, out)
            await update.message.reply_photo(photo=open(out_path, 'rb'))
        else:
            await update.message.reply_text("چهره‌ای پیدا نشد، عکس واضح‌تری بفرست.")

TOKEN = "YOUR_BOT_TOKEN"
app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(service_choice, pattern='^service_'))
app.add_handler(CallbackQueryHandler(color_choice, pattern='^(red|pink|purple|orange)$'))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
app.run_polling()
