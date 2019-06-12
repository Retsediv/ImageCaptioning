var file_to_process = "";

function get_caption(){
    return "Example caption"
}

function readURL(input) {
    console.log(input);
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            console.log("image: ", e.target.result);
            $('#image_upload_preview').attr('src', e.target.result);
        }

        reader.readAsDataURL(input.files[0]);
    }
}


$(document).ready(function(){
    $("#inputImage").change(function () {
        readURL(this);
    });

    $("#inputForm").submit(function (e) {
        e.preventDefault();

        let caption = get_caption()
        $("#image-caption-text").text(caption);

        $("#image-caption-block").show();
    });
});