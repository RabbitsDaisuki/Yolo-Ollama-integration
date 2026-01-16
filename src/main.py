import cv2
from .vision_module import PCBDetection as PCBs
from .llm_module import LLMDetection as LLMs
import logging
import time
from src import config
import threading

class SharedContext:
    visual_info = "Nothing detected yet"
    is_running = True
    is_listening = False

llm_data = SharedContext()

def LLM_Loop(llm_system):
    while llm_data.is_running:
        try:
            wav_file = llm_system._Take_user_sound()
            llm_data.is_listening = True

            if wav_file:
                user_text = llm_system._Transcribe_sound(wav_file)
                logging.info("LLM Status: Link Successfully")
            else:
                continue

            if not user_text:
                print("No voice detected retrying..")
                llm_data.is_listening = False
                continue

            new_vision = llm_data.visual_info

            try:
                llm_out = llm_system._Chat_With_Ollama(user_text, new_vision)
                logging.info("Take message from yolo")
            except:
                llm_out = llm_system._Chat_With_Ollama(user_text)
                logging.info("Not message from yolo")
        
            llm_system._Speak(llm_out)

            llm_data.is_listening = False

        except Exception as e:
            print(f"Thread Error: {e}")
            logging.error(f"Thread Error: {e}")
            time.sleep(1)

# --- GUI loop ---
def main():

    # ----- Initialize Logging -----
    logging.basicConfig(
        level = config.LOG_LEVEL,
        format = config.LOG_FORMAT,
        filename = str(config.LOG_FILE),
    )

    try:
        # --- Take Vision system ---
        pcb_system = PCBs() 
        # --- Take LLM System ---
        llm_system = LLMs()
    except Exception as e:
        print(f"Initialization Error: {e}")
        logging.error(f"Error: System can not take:{e}")
        return

    # --- Camera Loading ---
    Mac_cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not Mac_cap.isOpened():
        logging.error("System could not opened camera.")
        print("Error: No camera input, please check...")
        return
    else:
        llm_read = threading.Thread(
            target = LLM_Loop,
            args = (llm_system,)
        )
        # --- Ollama Status ---
        llm_read.daemon = True
        llm_read.start()
        print("System Running... Press 's' to get image , 'q' to exit.")
        logging.info("System status: Running")

    try:
        while True:
            start = time.time()
            ret, frame = Mac_cap.read()

            if not ret:
                print("Failed to receive frame, exiting...")
                logging.error("Could not generate frame, please check camera")
                break
            
            # --- Vision Loading ---
            annotated_frame, results = pcb_system.take_inference(frame)

            # --- Share data ---
            llm_data.visual_info = str(results)

            # -----show fps-----
            fps = 1/(time.time() - start)
            
            cv2.putText(annotated_frame,
                        f"FPS: {fps:.2f}, Status: AI Idle",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA
            )

            if llm_data.is_listening:
                cv2.putText(annotated_frame,
                        f"FPS: {fps:.2f}, Status: AI Busy...",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA
                        )
            else:
                cv2.putText(annotated_frame,
                        f"FPS: {fps:.2f}, Status: AI Listing...",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA
                        )


            
            # --- Show GUI ---
            cv2.imshow("Read_PCB", annotated_frame)

            input_key = cv2.waitKey(1) & 0xFF
            if input_key == ord('q'):
                print("System shutdown safely")
                logging.info("System shutdown safely")
                break
            
            elif input_key == ord('s'):
                pcb_system._img_save(annotated_frame)

    except KeyboardInterrupt:
        logging.info("System by pass")
        pass

    except Exception as e:
        print(f"An unexpected error occurred {e}")
        logging.error(f"System Error for {e}")
    
    finally:
        if Mac_cap.isOpened():
            Mac_cap.release()
        cv2.destroyAllWindows()
        print("System closed.")
        logging.info("Camera Status: Release successfully")

if __name__ == "__main__":
    main()



