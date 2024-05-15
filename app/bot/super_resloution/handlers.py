import requests

from aiogram import Router, F, types, Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from helpers.text_helper import get_text_from_config


class SuperResolutionStates(StatesGroup):
    start_super_resolution = State()
    send_pic = State()


super_resolution_router = Router()


@super_resolution_router.message(F.text.lower() == 'увеличить разрешение')
async def super_resolution_get_photo(message: types.Message, state: FSMContext):
    await message.answer(get_text_from_config('post_photo', block='RESPONSE'))
    await state.set_state(SuperResolutionStates.start_super_resolution)
    

@super_resolution_router.message(SuperResolutionStates.start_super_resolution, F.photo)
async def super_resolution_send_photo(message: types.Message, state: FSMContext, bot: Bot):
    path_to_photo = f'./temp/{message.photo[-1].file_unique_id}.jpg'
    await bot.download(message.photo[-1], destination=path_to_photo)
    await message.answer(get_text_from_config('api_proccess', block='RESPONSE'))

    proccess_file = requests.post('http://api:8000/super_resolution/', files={'file': open(path_to_photo, 'rb')})
    with open(path_to_photo, 'wb') as photo:
        photo.write(proccess_file.content)
    await message.answer_photo(photo=types.FSInputFile(path=path_to_photo))