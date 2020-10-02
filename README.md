# Sample project
You'll need to customize this quite a bit if you want to use it in your own project.
This will publish messages to an MQTT topic when motion is detected inside a zone (polygon).
It will only trigger when a person is detected for now.

### Sample docker-compose.yml file
```yaml
version: '3'
services:
  test_vision:
    image: test_vision:test
    restart: always
    volumes:
      - /mnt/NAS/http-static/snapshots/front-door:/data
    tmpfs:
      - /tmp-snap:size=100M
    environment:
      - DETECTION_ZONE={x,y;x,y}
      - MQTT_URL={IP ADDRESS:PORT}
      - MQTT_TOPIC=camera/front-door/motion-entry
      - MQTT_USERNAME=cameras
      - MQTT_PASSWORD={PASSWORD}
      - MQTT_PUBLISH_FREQUENCY=20000
      - DETECTION_WIDTH=320
      - DETECTION_FREQUENCY=6
      - MOTION_MIN_PERCENTAGE=0.0005
      - VIDEO_URL=rtsp://{url}
      - THUMBNAIL_URL={base_url_for_thumbnail}
```