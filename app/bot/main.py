import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart

from helpers.text_helper import get_text_from_config
from settings import BOT_TOKEN
from colorization.handlers import  colorization_router
from super_resloution.handlers import super_resolution_router
from denoising.handlers import denoising_router


bot = Bot(token=BOT_TOKEN)

dp = Dispatcher()
dp.include_router(colorization_router)
dp.include_router(super_resolution_router)
dp.include_router(denoising_router)


@dp.message(CommandStart())
async def command_start_handler(message: Message):
    text = get_text_from_config('greeting', block='RESPONSE')
    col_button_text = get_text_from_config('colorization', block='BUTTONS')
    deno_button_text = get_text_from_config('denoising', block='BUTTONS')
    super_button_text = get_text_from_config('super_resolution', block='BUTTONS')
    field_placeholder = get_text_from_config('start_placeholder', block='PLACEHOLDERS')

    buttons = [
        [KeyboardButton(text=col_button_text), KeyboardButton(text=deno_button_text)],
        [KeyboardButton(text=super_button_text)]
    ]

    keyboard = ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True, input_field_placeholder=field_placeholder)

    await message.answer(text, reply_markup=keyboard)
   
   


async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())