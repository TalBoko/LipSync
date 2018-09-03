
import time
import stomp
import urllib


class StompClient:
    def __init__(self):
        self.conn = stomp.Connection(host_and_ports=[("localhost", 61613)])
        self.conn.start()
        # NOTE: set correct user name and password
        self.conn.connect('admin', 'password', wait=True)
        # NOTE: chrBrad = target character name
        # NOTE: command string is url encoded need to change bml request id ==>> bml_??? in every request

        self.headers = dict()
        self.headers['ELVISH_SCOPE'] = 'DEFAULT_SCOPE'
        self.headers['MESSAGE_PREFIX'] = 'vrSpeak'
        self.conn.auto_content_length = False
        self.stomp_path_destination = '/topic/DEFAULT_SCOPE'

    def send_bml_speech(self, text):
        msg = "vrSpeak%20ChrBrad%20ALL%20bml_25722045%20%3C%3Fxml%20version%3D%221.0%22%20%3F%3E%3Cact%3E%3Cbml%3E%3C" \
              "speech%20type%3D%22text%2Fplain%22%3E{}%3C%2Fspeech%3E%3Chead%20type%3D%22NOD%22%20amount%3D%22.3%22" \
              "%2F%3E%3Cgaze%20target%3D%22ChrBrad%22%20sbm%3Ajoint-range%3D%22NECK%20EYES%22%2F%3E%3C%2Fbml%3E%3C" \
              "%2Fact%3E%20".format(text)
        self.conn.send(body=msg, headers=self.headers, destination=self.stomp_path_destination)
        time.sleep(1)

    def send_bml_face_example(self):
        msg = "vrSpeak ChrBrad+ALL+bml_25722044+%3c%3fxml+version%3d%221.0%22+%3f%3e%3cact%3e%3cbml%3e%3cface+au%3d" \
              "%2212%22+type%3d%22facs%22+side%3d%22BOTH%22+start%3d%220%22+end%3d%220.4%22+amount%3d%220.2%22%2f%3e" \
              "%3cface+au%3d%2227%22+type%3d%22facs%22+side%3d%22BOTH%22+start%3d%220%22+end%3d%220.4%22+amount%3d" \
              "%220.2%22%2f%3e%3c%2fbml%3e%3c%2fact%3e+ "
        # NOTE: face_decoded_message = "vrSpeak FuseCharacter ALL bml_25722044 <?xml version="1.0" ?><act><bml>
        # <face au="9" type="facs" side="BOTH" start="0" end="0.4" amount="0.2"/><face au="27" type="facs" side="BOTH"
        # start="0" end="0.4" amount="0.2"/></bml></act>"

        self.conn.send(body=msg, headers=self.headers, destination=self.stomp_path_destination)
        time.sleep(1)
        # NOTE: sadness
        msg = "vrSpeak%20ChrBrad%20ALL%20bml_25722045%20%3C%3Fxml%20version%3D%221.0%22%20%3F%3E%3Cact%3E%3Cbml%3E" \
            "%3Cface%20type%3D%22facs%22%20au%3D%221%22%20amount%3D%221%22%2F%3E%3Cface%20type%3D%22facs%22%20au%3D" \
            "%224%22%20amount%3D%221%22%2F%3E%3Cface%20type%3D%22facs%22%20au%3D%2215%22%20amount%3D%221%22%2F%3E%3C" \
            "%2Fbml%3E%3C%2Fact%3E%20 "
        self.conn.send(body=msg, headers=self.headers, destination=self.stomp_path_destination)
        time.sleep(1)

        # NOTE - source : http://smartbody.ict.usc.edu/HTML/documentation/SB/Face_8552847.html
        # happy msg example=
        # "vrSpeak%20ChrBrad%20ALL%20bml_25722045%20%3C%3Fxml%20version%3D%221.0%22%20%3F%3E%3Cact%3E%3Cbml%3E%3Cface
        # %20type%3D%22facs%22%20au%3D%226%22%20amount%3D%221%22%2F%3E%3Cface%20type%3D%22facs%22%20au%3D%2212%22
        # %20amount%3D%221%22%2F%3E%3C%2Fbml%3E%3C%2Fact%3E%20" self.conn.send(body=msg, headers=self.headers,
        # destination=self.stomp_path_destination) time.sleep(1) open mouth - doesn`t work

    def send_viseme_command(self,viseme):
        msg = "sb%20scene.getDiphoneManager%28%29.setPhonemesRealtime%28%22foo%22%2C%20%22{}%22%29".format(viseme)
        print(msg)
        self.conn.send(body=msg, headers=self.headers, destination=self.stomp_path_destination)

    def disconnect(self):
        self.conn.disconnect()

    def url_decode(self, msg):
        return urllib.unquote(msg).decode("utf8")

    def url_encode(self, msg):
        return urllib.quote(msg, safe='')


def main():
    stomp_client = StompClient()
    while True:
        stomp_client.send_viseme_command("aa")
        time.sleep(0.1)
        stomp_client.send_viseme_command("aa")
        time.sleep(0.1)
        stomp_client.send_viseme_command("th")
        time.sleep(0.1)
        stomp_client.send_bml_face_example()


if __name__ == '__main__':
    main()
