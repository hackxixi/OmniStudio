CREATE TABLE `model_params` (
	`model_key` text PRIMARY KEY NOT NULL,
	`params_json` text NOT NULL,
	`updated_at` integer NOT NULL
);
